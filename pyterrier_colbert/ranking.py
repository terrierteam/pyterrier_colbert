
import os
import torch
import pandas as pd
import pyterrier as pt
from pyterrier import tqdm
from pyterrier.transformer import TransformerBase
from typing import Union, Tuple
import random
from colbert.evaluation.load_model import load_model
from . import load_checkpoint
# monkeypatch to use our downloading version
import colbert.evaluation.loaders
colbert.evaluation.loaders.load_checkpoint = load_checkpoint
colbert.evaluation.loaders.load_model.__globals__['load_checkpoint'] = load_checkpoint
from colbert.modeling.inference import ModelInference
from colbert.evaluation.slow import slow_rerank
from colbert.indexing.loaders import get_parts, load_doclens
import colbert.modeling.colbert
from collections import defaultdict
import numpy as np
import pickle
from warnings import warn
from pyterrier_colbert.faiss_term_index import FaissNNTerm

class file_part_mmap:
    def __init__(self, file_path, file_doclens):
        self.dim = 128 # TODO
        
        self.doclens = file_doclens
        self.endpos = np.cumsum(self.doclens)
        self.startpos = self.endpos - self.doclens

        mmap_storage = torch.HalfStorage.from_file(file_path, False, sum(self.doclens) * self.dim)
        self.mmap = torch.HalfTensor(mmap_storage).view(sum(self.doclens), self.dim)
 
    def get_embedding(self, pid):
        startpos = self.startpos[pid]
        endpos = self.endpos[pid]
        return self.mmap[startpos:endpos,:]

class file_part_mem:
    def __init__(self, file_path, file_doclens):
        self.dim = 128 # TODO
        
        self.doclens = file_doclens
        self.endpos = np.cumsum(self.doclens)
        self.startpos = self.endpos - self.doclens

        self.mmap = torch.load(file_path)
        #print(self.mmap.shape)
 
    def get_embedding(self, pid):
        startpos = self.startpos[pid]
        endpos = self.endpos[pid]
        return self.mmap[startpos:endpos,:]


class Object(object):
    pass


from typing import List     


class re_ranker_mmap:
    def __init__(self, index_path, args, inference, verbose = False, memtype='mmap'):
        self.args = args
        self.doc_maxlen = args.doc_maxlen
        assert self.doc_maxlen > 0
        self.inference = inference
        self.dim = 128 #TODO
        self.verbose = verbose
    
        # Every pt file gets its own list of doc lengths
        self.part_doclens = load_doclens(index_path, flatten=False)
        assert len(self.part_doclens) > 0, "Did not find any indices at %s" % index_path
        # Local mmapped tensors with local, single file accesses
        self.part_mmap : List[file_part_mmap] = re_ranker_mmap._load_parts(index_path, self.part_doclens, memtype)
        
        # last pid (inclusive, e.g., the -1) in each pt file
        # the -1 is used in the np.searchsorted
        # so if each partition has 1000 docs, the array is [999, 1999, ...]
        # this helps us map from passage id to part (inclusive, explaning the -1)
        self.part_pid_end_offsets = np.cumsum([len(x) for x in self.part_doclens]) - 1
        
        # first pid (inclusive) in each pt file
        tmp = np.cumsum([len(x) for x in self.part_doclens])
        tmp[-1] = 0
        self.part_pid_begin_offsets = np.roll(tmp, 1)
        # [0, 1000, 2000, ...]
        self.part_pid_begin_offsets
    
    @staticmethod
    def _load_parts(index_path, part_doclens, memtype="mmap"):
        # Every pt file is loaded and managed independently, with local pids
        _, all_parts_paths, _ = get_parts(index_path)
        
        if memtype == "mmap":
            all_parts_paths = [ file.replace(".pt", ".store") for file in all_parts_paths ]
            mmaps = [file_part_mmap(path, doclens) for path, doclens in zip(all_parts_paths, part_doclens)]
        elif memtype == "mem":
            mmaps = [file_part_mem(path, doclens) for path, doclens in tqdm(zip(all_parts_paths, part_doclens), total=len(all_parts_paths), desc="Loading index shards to memory", unit="shard")]
        else:
            assert False, "Unknown memtype %s" % memtype
        return mmaps

    def get_embedding(self, pid):
        # In which pt file we need to look the given pid
        part_id = np.searchsorted(self.part_pid_end_offsets, pid)
        # calculate the pid local to the correct pt file
        local_pid = pid - self.part_pid_begin_offsets[part_id]
        # identify the tensor we look for
        disk_tensor = self.part_mmap[part_id].get_embedding(local_pid)
        doclen = disk_tensor.shape[0]
         # only here is there a memory copy from the memory mapped file 
        target = torch.zeros(self.doc_maxlen, self.dim)
        target[:doclen, :] = disk_tensor
        return target
    
    def get_embedding_copy(self, pid, target, index):
        # In which pt file we need to look the given pid
        part_id = np.searchsorted(self.part_pid_end_offsets, pid)
        # calculate the pid local to the correct pt file
        local_pid = pid - self.part_pid_begin_offsets[part_id]
        # identify the tensor we look for
        disk_tensor = self.part_mmap[part_id].get_embedding(local_pid)
        doclen = disk_tensor.shape[0]
         # only here is there a memory copy from the memory mapped file 
        target[index, :doclen, :] = disk_tensor
        return target
    
    def our_rerank(self, query, pids, gpu=True):
        colbert = self.args.colbert
        inference = self.inference

        Q = inference.queryFromText([query])
        if self.verbose:
            pid_iter = tqdm(pids, desc="lookups", unit="d")
        else:
            pid_iter = pids

        D_ = torch.zeros(len(pids), self.doc_maxlen, self.dim)
        for offset, pid in enumerate(pid_iter):
            self.get_embedding_copy(pid, D_, offset)

        if gpu:
            D_ = D_.cuda()

        scores = colbert.score(Q, D_).cpu()
        del(D_)
        return scores.tolist()

    def our_rerank_batched(self, query, pids, gpu=True, batch_size=1000):
        import more_itertools
        if len(pids) < batch_size:
            return self.our_rerank(query, pids, gpu=gpu)
        allscores=[]
        for group in more_itertools.chunked(pids, batch_size):
            batch_scores = self.our_rerank(query, group, gpu)
            allscores.extend(batch_scores)
        return allscores
        
        
    def our_rerank_with_embeddings(self, qembs, pids, weightsQ=None, gpu=True):
        """
        input: qid,query, docid, query_tokens, query_embeddings, query_weights 
        
        output: qid, query, docid, score
        """
        colbert = self.args.colbert
        inference = self.inference
        # default is uniform weight for all query embeddings
        if weightsQ is None:
            weightsQ = torch.ones(len(qembs))
        # make to 3d tensor
        Q = torch.unsqueeze(qembs, 0)
        if gpu:
            Q = Q.cuda()
        
        if self.verbose:
            pid_iter = tqdm(pids, desc="lookups", unit="d")
        else:
            pid_iter = pids

        D_ = torch.zeros(len(pids), self.doc_maxlen, self.dim)
        for offset, pid in enumerate(pid_iter):
            self.get_embedding_copy(pid, D_, offset)
        if gpu:
            D_ = D_.cuda()
        maxscoreQ = (Q @ D_.permute(0, 2, 1)).max(2).values.cpu()
        scores = (weightsQ*maxscoreQ).sum(1).cpu()
        return scores.tolist()

class ColBERTFactory():

    def __init__(self, 
            colbert_model : Union[str, Tuple[colbert.modeling.colbert.ColBERT, dict]], 
            index_root : str, 
            index_name : str,
            faiss_partitions=None,#TODO 100-
            memtype = "mem",
            gpu=True):
        
        args = Object()
        args.query_maxlen = 32
        args.doc_maxlen = 180
        args.dim = 128
        args.bsize = 128
        args.similarity = 'cosine'        
        args.dim = 128
        args.amp = True
        args.nprobe = 10
        args.part_range = None
        args.mask_punctuation = False
        args.partitions = faiss_partitions

        self.index_root = index_root
        self.index_name = index_name
        if index_root is None or index_name is None:
            warn("No index_root and index_name specified - no index ranking possible")
        else:
            self.index_path = os.path.join(index_root, index_name)
            docnos_file = os.path.join(self.index_path, "docnos.pkl.gz")
            if os.path.exists(docnos_file):
                with pt.io.autoopen(docnos_file, "rb") as f:
                    self.docid2docno = pickle.load(f)
                    # support reverse docno lookup in memory
                    self.docno2docid = { docno : docid for docid, docno in enumerate(self.docid2docno) }
                    self.docid_as_docno = False
            else:
                self.docid_as_docno = True

        try:
            import faiss
        except:
            warn("Faiss not installed. You cannot do retrieval")
        self.faiss_index_on_gpu = True
        if not gpu:
            self.faiss_index_on_gpu = False
            warn("Gpu disabled, YMMV")
            import colbert.parameters
            colbert.parameters.DEVICE = torch.device("cpu")
        if isinstance (colbert_model, str):
            args.checkpoint = colbert_model
            args.colbert, args.checkpoint = load_model(args)
        else:
            assert isinstance(colbert_model, tuple)
            args.colbert, args.checkpoint = colbert_model
            from colbert.modeling.colbert import ColBERT
            assert isinstance(args.colbert, ColBERT)
            assert isinstance(args.checkpoint, dict)
            
        args.inference = ModelInference(args.colbert, amp=args.amp)
        self.args = args

        self.memtype = memtype

        #we load this lazily
        self.rrm = None
        self.faiss_index = None
        
    def _rrm(self):
        """
        Returns an instance of the re_ranker_mmap class.
        Only one is created, if necessary.
        """

        if self.rrm is not None:
            return self.rrm
        print("Loading reranking index, memtype=%s" % self.memtype)
        self.rrm = re_ranker_mmap(
            self.index_path, 
            self.args, 
            self.args.inference, 
            verbose=self.verbose, 
            memtype=self.memtype)
        return self.rrm
        
    def nn_term(self, df=False):
        """
        Returns an instance of the FaissNNTerm class, which provides statistics about terms
        """
        if self.faissnn is not None:
            return self.faissnn
        from colbert.ranking.faiss_term_index import FaissNNTerm
        #TODO accept self.args.inference as well
        self.faissnn = FaissNNTerm(
            self.args.colbert,
            self.index_root,
            self.index_name,
            faiss_index = self._faiss_index(),
            df=df)
        return self.faissnn

    def query_encoder(self, detach=True) -> TransformerBase:
        """
        Returns a transformer that can encode queries using ColBERT's model.
        input: qid, query
        output: qid, query, query_embs, query_toks,
        """
        def _encode_query(row):
            with torch.no_grad():
                Q, ids, masks = self.args.inference.queryFromText([row.query], bsize=512, with_ids=True)
                if detach:
                    Q = Q.cpu()
                return pd.Series([Q[0], ids[0]])
            
        def row_apply(df):
            df[["query_embs", "query_toks"]] = df.apply(_encode_query, axis=1)
            return df
        
        return pt.apply.generic(row_apply)

    def _faiss_index(self):
        """
        Returns an instance of the Colbert FaissIndex class, which provides nearest neighbour information
        """
        from colbert.indexing.faiss import get_faiss_index_name
        from colbert.ranking.faiss_index import FaissIndex
        if self.faiss_index is not None:
            return self.faiss_index
        faiss_index_path = get_faiss_index_name(self.args)
        faiss_index_path = os.path.join(self.index_path, faiss_index_path)
        if not os.path.exists(faiss_index_path):
            raise ValueError("No faiss index found at %s" % faiss_index_path)
        self.faiss_index = FaissIndex(self.index_path, faiss_index_path, self.args.nprobe, self.args.part_range)
        # ensure the faiss_index is transferred to GPU memory for speed
        import faiss
        if self.faiss_index_on_gpu:
            self.faiss_index.faiss_index = faiss.index_cpu_to_all_gpus(self.faiss_index.faiss_index)
        return self.faiss_index

    def set_retrieve(self, batch=False, query_encoded=False, faiss_depth=1000, verbose=False, docnos=False) -> TransformerBase:
        #input: qid, query
        #OR
        #input: qid, query, query_embs, query_toks, query_weights

        #output: qid, query, docid, [docno]
        #OR
        #output: qid, query, query_embs, query_toks, query_weights, docid, [docno]
        
        assert not batch
        faiss_index = self._faiss_index()
        
        # this is when queries have NOT already been encoded
        def _single_retrieve(queries_df):
            rtr = []
            iter = queries_df.itertuples()
            iter = tqdm(iter, unit="q")  if verbose else iter
            for row in iter:
                qid = row.qid
                query = row.query
                with torch.no_grad():
                    Q, ids, masks = self.args.inference.queryFromText([query], bsize=512, with_ids=True)
                Q_f = Q[0:1, :, :]
                all_pids = faiss_index.retrieve(faiss_depth, Q_f, verbose=verbose)
                Q_cpu = Q[0, :, :].cpu()
                for passage_ids in all_pids:
                    if verbose:
                        print("qid %s retrieved docs %d" % (qid, len(passage_ids)))
                    for pid in passage_ids:
                        rtr.append([qid, query, pid, ids[0], Q_cpu])
            rtrDf = pd.DataFrame(rtr, columns=["qid","query",'docid','query_toks','query_embs'] )
            if docnos:
                rtrDf = self._add_docnos(rtrDf)
            return rtrDf

        # this is when queries have already been encoded
        def _single_retrieve_qembs(queries_df):
            rtr = []
            query_weights = "query_weights" in queries_df.column
            iter = queries_df.itertuples()
            iter = tqdm(iter, unit="q") if verbose else iter
            for row in iter:
                qid = row.qid
                embs = row.query_embs
                Q_f = torch.unsqueeze(embs, 0)
                all_pids = faiss_index.retrieve(faiss_depth, Q_f, verbose=verbose)
                for passage_ids in all_pids:
                    if verbose:
                        print("qid %s retrieved docs %d" % (qid, len(passage_ids)))
                    for pid in passage_ids:
                        if query_weights:
                           rtr.append([qid, row.query, pid, row.query_toks, row.query_embs, row.query_weights])
                        else:
                           rtr.append([qid, row.query, pid, row.query_toks, row.query_embs])
            rtrDf = pd.DataFrame(rtr, columns=["qid","query",'docid','query_toks','query_embs'])
            if docnos:
                rtrDf = self._add_docnos(rtrDf)
            return rtrDf
        
        return pt.apply.generic(_single_retrieve_qembs if query_encoded else _single_retrieve)

    def text_scorer(self, query_encoded=False, doc_attr="text", verbose=False) -> TransformerBase:
        """
        Returns a transformer that uses ColBERT model to score the *text* of documents.
        """
        #input: qid, query, docno, text
        #OR
        #input: qid, query, query_embs, query_toks, query_weights, docno, text

        #output: qid, query, docno, score

        assert not query_encoded
        def _text_scorer(queries_and_docs):
            groupby = queries_and_docs.groupby("qid")
            rtr=[]
            with torch.no_grad():
                for qid, group in tqdm(groupby, total=len(groupby), unit="q") if verbose else groupby:
                    query = group["query"].values[0]
                    ranking = slow_rerank(self.args, query, group["docno"].values, group[doc_attr].values.tolist())
                    for rank, (score, pid, passage) in enumerate(ranking):
                            rtr.append([qid, query, pid, score, rank])          
            return pd.DataFrame(rtr, columns=["qid", "query", "docno", "score", "rank"])

        return pt.apply.generic(_text_scorer)

    def _add_docids(self, df):
        if self.docid_as_docno:
            df["docid"] = df["docno"].astype('int64')
        else:
            df["docid"] = df["docno"].apply(lambda docno : self.docno2docid[docno])
        return df

    def _add_docnos(self, df):
        if self.docid_as_docno:
            df["docno"] = df["docid"].astype('str')
        else:
            df["docno"] = df["docid"].apply(lambda docid : self.docid2docno[docid])
        return df

    def index_scorer(self, query_encoded=False, add_ranks=False) -> TransformerBase:
        """
        Returns a transformer that uses the ColBERT index to perform scoring of documents to queries 
        """
        #input: qid, query, docno, [docid] 
        #OR
        #input: qid, query, query_embs, query_toks, query_weights, docno

        #output: qid, query, docno, score

        rrm = self._rrm()

        def rrm_scorer(qid_group):
            qid_group = qid_group.copy()
            if "docid" not in qid_group.columns:
                qid_group = self._add_docids(qid_group)
            qid_group.sort_values("docid", inplace=True)
            docids = qid_group["docid"].values
            scores = rrm.our_rerank_batched(qid_group.iloc[0]["query"], docids)
            qid_group["score"] = scores
            if add_ranks:
                return pt.model.add_ranks(qid_group)
            return qid_group

        def rrm_scorer_query_embs(qid_group):
            qid_group = qid_group.copy()
            if "docid" not in qid_group.columns:
                qid_group = self._add_docids(qid_group)
            qid_group.sort_values("docid", inplace=True)
            docids = qid_group["docid"].values
            weights = None
            if "query_weights" in qid_group.columns:
                weights = qid_group.iloc[0].query_weights
            #TODO batching
            scores = rrm.our_rerank_with_embeddings(qid_group.iloc[0]["query_embs"], docids, weights)
            qid_group["score"] = scores
            if add_ranks:
                return pt.model.add_ranks(qid_group)
            return qid_group

        if query_encoded:
            return pt.apply.by_query(rrm_scorer_query_embs) 
        return pt.apply.by_query(rrm_scorer) 

    def end_to_end(self) -> TransformerBase:
        """
        Returns a transformer composition that uses a ColBERT FAISS index to retrieve documents, followed by a ColBERT index 
        to perform accurate scoring of the retrieved documents. Equivalent to `colbertfactory.set_retrieve() >> colbertfactory.index_scorer()`.
        """
        #input: qid, query, 
        #output: qid, query, docno, score
        return self.set_retrieve() >> self.index_scorer()

    def prf(pytcolbert, reranker, fb_docs=3, fb_embs=10, beta=1.0, k=24) -> TransformerBase:
        """
        Returns a pipeline for ColBERT PRF, either as a ranker, or a re-ranker. Final ranking is limited to 1000 docs.
    
        Parameters:
         - reranker(bool): Whether to rerank the initial documents, or to perform a new set retrieve to gather new documents.
         - fb_docs(int): Number of passages to use as feedback. Defaults to 3. 
         - k(int): Number of clusters to apply on the embeddings of the top K documents. Defaults to 24.
         - fb_terms(int): Number of expansion embeddings to add to the query. Defaults to 10.
         - beta(float): Weight of the new embeddings compared to the original emebddings. Defaults to 1.0.

        Reference:
        
        X. Wang, C. Macdonald, N. Tonellotto, I. Ounis. Pseudo-Relevance Feedback for Multiple Representation Dense Retrieval. 
        In Proceedings of ICTIR 2021.
        
        """
        dense_e2e = pytcolbert.set_retrieve() >> pytcolbert.index_scorer(query_encoded=True, add_ranks=True)
        if reranker:
            prf_pipe = (
                dense_e2e  
                >> ColbertPRF(pytcolbert, k=k, fb_docs=fb_docs, fb_embs=fb_embs, beta=beta, return_docs=True)
                >> (pytcolbert.index_scorer(query_encoded=True, add_ranks=True) %1000)
            )
        else:
            prf_pipe = (
                dense_e2e  
                >> ColbertPRF(pytcolbert, k=k, fb_docs=fb_docs, fb_embs=fb_embs, beta=beta, return_docs=False)
                >> pytcolbert.set_retrieve(query_encoded=True)
                >> (pytcolbert.index_scorer(query_encoded=True, add_ranks=True) % 1000)
            )
        return prf_pipe

    def explain_doc(self, query : str, docno : str):
        """
        Provides a diagram explaining the interaction between a query and a given docno
        """
        raise NotImplementedError()
        pid = self.docno2docid[docno]
        embsD = self.get_embedding(pid)
        return self._explain(query, embsD, idsD)

    def explain_text(self, query : str, document : str):
        """
        Provides a diagram explaining the interaction between a query and the text of a document
        """
        embsD, idsD = self.args.inference.docFromText([document], with_ids=True)
        return self._explain(query, embsD, idsD)
    
    def _explain(self, query, embsD, idsD):
        embsQ, idsQ, masksQ = self.args.inference.queryFromText([query], with_ids=True)

        interaction = (embsQ[0] @ embsD[0].T).cpu().numpy().T
        
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        tokenmap = {"[unused1]" : "[D]", "[unused0]" : "[Q]"}

        fig = plt.figure(figsize=(4, 12)) 
        gs = GridSpec(2, 1, height_ratios=[1, 20]) 

        ax1=fig.add_subplot(gs[0])
        ax2=fig.add_subplot(gs[1])
        
        ax2.matshow(interaction, cmap=plt.cm.Blues)
        qtokens = self.args.inference.query_tokenizer.tok.convert_ids_to_tokens(idsQ[0])
        dtokens = self.args.inference.query_tokenizer.tok.convert_ids_to_tokens(idsD[0])
        qtokens = [tokenmap[t] if t in tokenmap else t for t in qtokens]
        dtokens = [tokenmap[t] if t in tokenmap else t for t in dtokens]

        ax2.set_xticks(range(32), minor=False)
        ax2.set_xticklabels(qtokens, rotation=90)
        ax2.set_yticks(range(len(idsD[0])))
        ax2.set_yticklabels(dtokens)
        ax2.set_anchor("N")

        contributions=[]
        for i in range(32):
            maxpos = np.argmax(interaction[:,i])
            plt.text(i-0.25, maxpos+0.1, "X", fontsize=5)
            contributions.append(interaction[maxpos,i])

        ax1.bar([0.5 + i for i in range(0,32)], contributions)
        ax1.set_xticklabels([])
        fig.tight_layout()
        #fig.subplots_adjust(hspace=-0.37)
        fig.show()

from pyterrier.transformer import TransformerBase
import pandas as pd

class ColbertPRF(TransformerBase):
    def __init__(self, pytcfactory, k, fb_embs, num_docs, beta=1, r = 42, return_docs = False, fb_docs=10,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = k
        self.fb_embs = fb_embs
        self.beta = beta
        self.return_docs = return_docs
        self.fb_docs = fb_docs
        self.pytcfactory = pytcfactory
        self.fnt = pytcfactory.nn_term(df=True)
        self.r = r
        import torch
        import numpy as np
        num_docs = len(self.fnt.doclens)
        self.idfdict = {}
        for tid in pt.tqdm(range(self.fnt.inference.query_tokenizer.tok.vocab_size)):
            df = self.fnt.getDF_by_id(tid)
            idfscore = np.log((1.0+num_docs)/(df+1))
            self.idfdict[tid] = idfscore
        assert self.k > self.fb_embs ,"fb_embs should be smaller than number of clusters"
        self._init_clustering()

    def _init_clustering(self):
        import sklearn
        from packaging.version import Version
        from warnings import warn
        if Version(sklearn.__version__) > Version('0.23.2'):
            warn("You have sklearn version %s - sklearn KMeans clustering changed in 0.24, so performance may differ from those reported in the ICTIR 2021 paper."
            "See also https://github.com/scikit-learn/scikit-learn/issues/19990" % str(sklearn.__version__))

    def _get_centroids(self, prf_embs):
        from sklearn.cluster import KMeans
        kmn = KMeans(self.k, random_state=self.r)
        kmn.fit(prf_embs)
        return kmn.cluster_centers_
        
    def transform_query(self, topic_and_res : pd.DataFrame) -> pd.DataFrame:
        from sklearn.cluster import KMeans
        topic_and_res = topic_and_res.sort_values('rank')
        prf_embs = torch.cat([self.pytcfactory.rrm.get_embedding(docid) for docid in topic_and_res.head(self.fb_docs).docid.values])
        centroids = self._get_centroids(prf_embs)
        
        emb_and_score = []
        for cluster in range(self.k):
            centroid = np.float32( centroids[cluster] )
            tok2freq = self.nn_term.get_nearest_tokens_for_emb(self.fnt, centroid)
            if len(tok2freq) == 0:
                continue
            most_likely_tok = max(tok2freq, key=tok2freq.get)
            tid = self.fnt.inference.query_tokenizer.tok.convert_tokens_to_ids(most_likely_tok)      
            emb_and_score.append( (centroid, most_likely_tok, tid, self.idfdict[tid]) ) 
        
        sorted_by_second = sorted(emb_and_score, key=lambda tup: -tup[3])
        
        toks=[]
        scores=[]
        exp_embds = []
        for i in range(min(self.fb_embs, len(sorted_by_second))):
            emb, tok, tid, score = sorted_by_second[i]
            toks.append(tok)
            scores.append(score)
            exp_embds.append(emb)
        
        first_row = topic_and_res.iloc[0]
        newemb = torch.cat([
            first_row.query_embs, 
            torch.Tensor(exp_embds)])
        
        weights = torch.cat([ 
            torch.ones(len(first_row.query_embs)),
            self.beta * torch.Tensor(scores)]
        )
        
        rtr = pd.DataFrame([
            [first_row.qid, 
             first_row.docno,
             first_row.query, 
             newemb, 
             toks, 
             weights ]
            ],
            columns=["qid","docno", "query", "query_embs", "query_toks", "query_weights"])
        return rtr

    def transform(self, topics_and_docs : pd.DataFrame) -> pd.DataFrame:
        # validation of the input
        required = ["qid", "query", "docno", "query_embs", "rank"]
        for col in required:
            assert col in topics_and_docs.columns
        
        #restore the docid column if missing
        if "docid" not in topics_and_docs:
            topics_and_docs = self.pytcfactory.add_docids(topics_and_docs)
        
        rtr = []
        for qid, res in topics_and_docs.groupby("qid"):
            new_query_df = self.transform_query(res)     
            if self.return_docs:
                new_query_df = res[["qid", "docno", "docid"]].merge(new_query_df, on=["qid"])
                
                new_query_df = new_query_df.rename(columns={'docno_x':'docno'})
            rtr.append(new_query_df)
        return pd.concat(rtr)
