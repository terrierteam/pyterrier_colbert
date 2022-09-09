import os
import time
import random
import torch
import numpy as np
from collections import defaultdict
from multiprocessing import Pool
from colbert.modeling.inference import ModelInference

from colbert.utils.utils import print_message, flatten, batch
from colbert.indexing.loaders import get_parts

from colbert.evaluation.loaders import load_colbert
from colbert.indexing.faiss import get_faiss_index_name
from colbert.indexing.loaders import load_doclens


from tqdm import tqdm

def load_tokenids(directory):
    parts, _, _ = get_parts(directory)

    tokenids_filenames = [os.path.join(directory, str(part) + ".tokenids") for part in parts]
    all_tokenids = torch.cat([torch.load(filename) for filename in tokenids_filenames])

    return all_tokenids

class Object(object):
    pass

class FaissNNTerm():

    def __init__(self, colbert, index_root, index_name, nprobe=10, partitions=None, part_range=None, query_maxlen=32, faiss_index=None, cf=True, df=False, mask_punctuation=False):
        if type(colbert) == str:
            args = Object()
            args.checkpoint = colbert
            args.dim = 128
            args.similarity = 'cosine'
            args.query_maxlen = 32
            args.doc_maxlen = 180
            args.mask_punctuation = mask_punctuation
            self.colbert, args.checkpoint = load_colbert(args)
        else:
            self.colbert = colbert

        index_path = os.path.join(index_root, index_name)
        if faiss_index is None:
            faiss_name = None 
            if faiss_name is not None:
                faiss_index_path = os.path.join(index_path, faiss_name)
            else:
                args = Object()
                args.partitions = partitions
                faiss_index_path = os.path.join(index_path, get_faiss_index_name(args))
            print_message("#> Loading the FAISS index from", faiss_index_path, "..")

            faiss_part_range = os.path.basename(faiss_index_path).split('.')[-2].split('-')

            if len(faiss_part_range) == 2:
                faiss_part_range = range(*map(int, faiss_part_range))
                assert part_range[0] in faiss_part_range, (part_range, faiss_part_range)
                assert part_range[-1] in faiss_part_range, (part_range, faiss_part_range)
            else:
                faiss_part_range = None

            self.part_range = part_range
            self.faiss_part_range = faiss_part_range
            from colbert.ranking.faiss_index import FaissIndex
            self.faiss_index = FaissIndex(index_path, faiss_index_path, nprobe, faiss_part_range)

        else:
            self.faiss_index = faiss_index
            self.faiss_index.nprobe = nprobe
        
        self.inference = ModelInference(self.colbert)
        self.skips = set(self.inference.query_tokenizer.tok.special_tokens_map.values())
        print_message("#> Building the emb2tid mapping..")
        self.emb2tid = load_tokenids(index_path)
        print(len(self.emb2tid))
        
        self.tok = self.inference.query_tokenizer.tok
        vocab_size = self.tok.vocab_size
        if cf:
            cfs_file = os.path.join(index_path, "cfs.stats")
            if os.path.exists(cfs_file):
                self.lookup = torch.load(cfs_file)
            else:
                print("Computing collection frequencies")
                self.lookup = torch.zeros(vocab_size, dtype=torch.int64)
                indx, cnt = self.emb2tid.unique(return_counts=True)
                self.lookup[indx] += cnt
                print("Done")
                torch.save(self.lookup, cfs_file)
        
        print("Loading doclens")
        part_doclens = load_doclens(index_path, flatten=False)
        import numpy as np
        self.doclens = np.concatenate([np.array(part) for part in part_doclens])
        self.num_docs = len(self.doclens)
        self.end_offsets = np.cumsum(self.doclens)
        if df:
            dfs_file = os.path.join(index_path, "dfs.stats")
            if os.path.exists(dfs_file):
                self.dfs = torch.load(dfs_file)
            else:
                dfs = torch.zeros(vocab_size, dtype=torch.int64)
                offset = 0
                for doclen in tqdm(self.doclens, unit="d", desc="Computing document frequencies"):
                    tids= torch.unique(self.emb2tid[offset:offset+doclen])
                    dfs[tids] += 1
                    offset += doclen
                self.dfs = dfs
                torch.save(dfs, dfs_file)

    
    def get_tokens_for_doc(self, pid):
        """
        Returns the actual indexed tokens within a given document
        """
        end_offset = self.end_offsets[pid]
        start_offset = end_offset - self.doclens[pid]
        return self.emb2tid[start_offset:end_offset]
    
    def get_nearest_tokens_for_embs(self, embs : np.array, k=10, low_tf=0):
        """
            Returns the most related terms for each of a number of given embeddings
        """
        from collections import defaultdict
        assert len(embs.shape) == 2
        
        _, ids = self.faiss_index.faiss_index.search(embs, k=k)
        
        rtrs=[]
        for id_set in ids:
            id2freq = defaultdict(int)
            for id in id_set:
                id2freq[self.emb2tid[id].item()] += 1
            rtr = {}
            for t, freq in sorted(id2freq.items(), key=lambda item: -1* item[1]):
                if freq <= low_tf:
                    continue
                token = self.inference.query_tokenizer.tok.decode([t])
                if "[unused" in token or token in self.skips:
                    continue
                rtr[token] = freq
            rtrs.append(rtr)
        return rtrs

    def get_nearest_tokens_for_emb(self, emb : np.array, k=10, low_tf=0):
        """
            Returns the most related terms for one given embedding
        """
        return self.get_nearest_tokens_for_embs(np.expand_dims(emb, axis=0), k=k, low_tf=low_tf)[0]

    def getCTF(self, term):
        """
            Returns the collection frequency of a given token string.
            It is assumed to be a known token(piece) to BERT, and thus not
            required tokenisation
        """
        id = self.tok.convert_tokens_to_ids(term)
        return self.lookup[id].item()

    def getCTF_by_id(self, tid):
        """
            Returns the collection frequency of a given token id
        """
        return self.lookup[tid].item()

    def getDF(self, term):
        """
            Returns the document frequency of a given token string.
            It is assumed to be a known token(piece) to BERT, and thus not
            required tokenisation
        """
        id = self.tok.convert_tokens_to_ids(term)
        return self.dfs[id].item()

    def getDF_by_id(self, tid):
        """
            Returns the document frequency of a given token id
        """
        return self.dfs[tid].item()
    
    def display_nn_terms(self, q, k=10, low_tf=0, n=0, by_term=True):
        """
            Displays the most related terms for each query
        """

        import numpy as np
        with torch.no_grad():
            input_ids, attention_mask = self.inference.query_tokenizer.tensorize([q])
            qembs = self.inference.query(input_ids, attention_mask)
            qembs = torch.nn.functional.normalize(qembs, p=2, dim=2)
        #ids = self.faiss_index.queries_to_embedding_ids(k, qembs)
        scores, ids = self.faiss_index.faiss_index.search(qembs[0].cpu().numpy(), k=k)
        if by_term:
            
            for (tScores, id_set, src_id) in zip(scores, ids, input_ids[0]):
                token = self.inference.query_tokenizer.tok.decode([src_id])
                print(token)
                for embid, score in zip(id_set, tScores):
                    tid = self.emb2tid[embid].item()
                    did = self.faiss_index.emb2pid[embid].item()
                    token = self.inference.query_tokenizer.tok.decode([tid])
                    print("\t%s (embid %d tid %d did %d) %0.5f" % (token, embid, tid, did, score))
        else:
            id2freq = defaultdict(int)
            for id_set in ids:
                for id in id_set:
                    id2freq[self.emb2tid[id].item()] += 1
            rtr = q
            count=0
            for t, freq in sorted(id2freq.items(), key=lambda item: -1* item[1]):
                if freq <= low_tf:
                    continue
                token = self.inference.query_tokenizer.tok.decode([t])
                if "[unused" in token or token in self.skips:
                    continue
                rtr += "\n\t" + token + " ("+str(t)+") x " + str(freq) + " "
                count+= 1
                if n > 0 and count == n:
                    break
            print(rtr)
            

