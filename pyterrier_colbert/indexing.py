import os
from pyterrier_colbert.ranking import ColBERTFactory
import numpy as np
from pyterrier.transformer import TransformerBase
from colbert.indexing.faiss import index_faiss
from colbert.indexing.loaders import load_doclens
from pyterrier.transformer import TransformerBase
from pyterrier.transformer import IterDictIndexerBase
import os
import ujson
import random
import copy
import queue
import math
from colbert.utils.parser import Arguments
import colbert.utils.distributed as distributed

from colbert.utils.utils import create_directory

import os
import time
import torch
import ujson
import numpy as np

import itertools
import threading
import queue

from colbert.modeling.inference import ModelInference
from colbert.evaluation.loaders import load_colbert
from . import load_checkpoint
# monkeypatch to use our downloading version
import colbert.evaluation.loaders
colbert.evaluation.loaders.load_checkpoint = load_checkpoint
colbert.evaluation.loaders.load_model.__globals__['load_checkpoint'] = load_checkpoint
from colbert.utils.utils import print_message
import pickle
from colbert.indexing.index_manager import IndexManager
from warnings import warn


class CollectionEncoder():
    def __init__(self, args, process_idx, num_processes):
        self.args = args
        self.collection = args.collection
        self.process_idx = process_idx
        self.num_processes = num_processes
        self.iterator = self._initialize_iterator()

        assert 0.5 <= args.chunksize <= 128.0
        max_bytes_per_file = args.chunksize * (1024*1024*1024)

        max_bytes_per_doc = (self.args.doc_maxlen * self.args.dim * 2.0)

        minimum_subset_size = 10_000
        maximum_subset_size = max_bytes_per_file / max_bytes_per_doc
        maximum_subset_size = max(minimum_subset_size, maximum_subset_size)
        self.possible_subset_sizes = [int(maximum_subset_size)]

        self.print_main("#> Local args.bsize =", args.bsize)
        self.print_main("#> args.index_root =", args.index_root)
        self.print_main(f"#> self.possible_subset_sizes = {self.possible_subset_sizes}")

        self._load_model()
        self.indexmgr = IndexManager(args.dim)

    def _initialize_iterator(self):
        return open(self.collection)

    def _saver_thread(self):
        for args in iter(self.saver_queue.get, None):
            self._save_batch(*args)

    def _load_model(self):
        self.colbert, self.checkpoint = load_colbert(self.args, do_print=(self.process_idx == 0))
        self.colbert = self.colbert.cuda()
        self.colbert.eval()

        self.inference = ModelInference(self.colbert, amp=self.args.amp)

    def encode(self):
        self.saver_queue = queue.Queue(maxsize=3)
        thread = threading.Thread(target=self._saver_thread)
        thread.start()

        t0 = time.time()
        local_docs_processed = 0

        for batch_idx, (offset, lines, owner) in enumerate(self._batch_passages(self.iterator)):
            if owner != self.process_idx:
                continue

            t1 = time.time()
            batch = self._preprocess_batch(offset, lines)
            embs, doclens = self._encode_batch(batch_idx, batch)

            t2 = time.time()
            self.saver_queue.put((batch_idx, embs, offset, doclens))

            print(len(lines))

            t3 = time.time()
            local_docs_processed += len(lines)
            overall_throughput = compute_throughput(local_docs_processed, t0, t3)
            this_encoding_throughput = compute_throughput(len(lines), t1, t2)
            this_saving_throughput = compute_throughput(len(lines), t2, t3)

            self.print(f'#> Completed batch #{batch_idx} (starting at passage #{offset}) \t\t'
                          f'Passages/min: {overall_throughput} (overall), ',
                          f'{this_encoding_throughput} (this encoding), ',
                          f'{this_saving_throughput} (this saving)')
        self.saver_queue.put(None)

        self.print("#> Joining saver thread.")
        thread.join()

    def _batch_passages(self, fi):
        """
        Must use the same seed across processes!
        """
        np.random.seed(0)

        offset = 0
        for owner in itertools.cycle(range(self.num_processes)):
            batch_size = np.random.choice(self.possible_subset_sizes)

            L = [line for _, line in zip(range(batch_size), fi)]

            if len(L) == 0:
                break  # EOF

            yield (offset, L, owner)
            offset += len(L)

            if len(L) < batch_size:
                break  # EOF

        self.print("[NOTE] Done with local share.")

        return

    def _preprocess_batch(self, offset, lines):
        endpos = offset + len(lines)

        batch = []

        for line_idx, line in zip(range(offset, endpos), lines):
            line_parts = line.strip().split('\t')

            pid, passage, *other = line_parts

            assert len(passage) >= 1

            if len(other) >= 1:
                title, *_ = other
                passage = title + ' | ' + passage

            batch.append(passage)

            assert pid == 'id' or int(pid) == line_idx

        return batch

    def _encode_batch(self, batch_idx, batch):
        with torch.no_grad():
            embs = self.inference.docFromText(batch, bsize=self.args.bsize, keep_dims=False)
            assert type(embs) is list
            assert len(embs) == len(batch)

            local_doclens = [d.size(0) for d in embs]
            embs = torch.cat(embs)

        return embs, local_doclens

    def _save_batch(self, batch_idx, embs, offset, doclens):
        start_time = time.time()

        output_path = os.path.join(self.args.index_path, "{}.pt".format(batch_idx))
        output_sample_path = os.path.join(self.args.index_path, "{}.sample".format(batch_idx))
        doclens_path = os.path.join(self.args.index_path, 'doclens.{}.json'.format(batch_idx))

        # Save the embeddings.
        print(output_path)
        self.indexmgr.save(embs, output_path)
        self.indexmgr.save(embs[torch.randint(0, high=embs.size(0), size=(embs.size(0) // 20,))], output_sample_path)

        # Save the doclens.
        with open(doclens_path, 'w') as output_doclens:
            ujson.dump(doclens, output_doclens)

        throughput = compute_throughput(len(doclens), start_time, time.time())
        self.print_main("#> Saved batch #{} to {} \t\t".format(batch_idx, output_path),
                        "Saving Throughput =", throughput, "passages per minute.\n")

    def print(self, *args):
        print_message("[" + str(self.process_idx) + "]", "\t\t", *args)

    def print_main(self, *args):
        if self.process_idx == 0:
            self.print(*args)


def compute_throughput(size, t0, t1):
    throughput = size / (t1 - t0) * 60

    if throughput > 1000 * 1000:
        throughput = throughput / (1000*1000)
        throughput = round(throughput, 1)
        return '{}M'.format(throughput)

    throughput = throughput / (1000)
    throughput = round(throughput, 1)
    return '{}k'.format(throughput)


class Object(object):
  pass

class CollectionEncoder_Generator(CollectionEncoder):

    def __init__(self, prepend_title, *args):#, prepend_title=False):
        super().__init__(*args)
        self.prepend_title = prepend_title

    def _initialize_iterator(self):
      return self.args.generator

    def _preprocess_batch(self, offset, lines):
        endpos = offset + len(lines)

        batch = []
        prepend_title = self.prepend_title

        for line_idx, line in zip(range(offset, endpos), lines):
#            pid = line["docid"]
            passage = line["text"]
            if prepend_title:
                title = line["title"]
                passage = title + ' | ' + passage
            assert len(passage) >= 1

            batch.append(passage)

#            assert pid == 'id' or int(pid) == line_idx

        return batch


class ColBERTIndexer(IterDictIndexerBase):
    def __init__(self, checkpoint, index_root, index_name, chunksize):
        args = Object()
        args.similarity = 'cosine'
        args.dim = 128
        args.query_maxlen = 32
        args.doc_maxlen = 180
        args.mask_punctuation = False
        args.checkpoint = checkpoint
        args.bsize = 128
        args.collection = None
        args.amp = False
        args.index_root = index_root
        args.index_name = index_name
        args.chunksize = chunksize
        args.rank = -1
        args.index_path = os.path.join(args.index_root, args.index_name)
        args.input_arguments = copy.deepcopy(args)
        args.nranks, args.distributed = distributed.init(args.rank)
        self.saver_queue = queue.Queue(maxsize=3)
        args.partitions = 100
        args.prepend_title = False
        self.args = args
        self.args.sample = None
        self.args.slices = 1

        assert self.args.slices >= 1
        assert self.args.sample is None or (0.0 < self.args.sample <1.0), self.args.sample

    def ranking_factory(self, memtype="mem"):
        return ColBERTFactory(
            (self.colbert, self.checkpoint),
            self.args.index_root,
            self.args.index_name,
            self.args.partitions,
            memtype
        )

    def index(self, iterator, prepend_title=False, num_docs=None):
        from timeit import default_timer as timer
        starttime = timer()
        maxdocs = 100
        assert not os.path.exists(self.args.index_path), self.args.index_path
        docnos=[]
        docid=0
        def convert_gen(iterator):
            import pyterrier as pt
            nonlocal docnos
            nonlocal docid
            if num_docs is not None:
                iterator = pt.tqdm(iterator, total=num_docs, desc="encoding", unit="d")
            for l in iterator:
                l["docid"] = docid
                docnos.append(l['docno'])
                docid+=1
                yield l              
        self.args.generator = convert_gen(iterator)
        self.args.prepend_title = prepend_title
        print("index ", self.args.prepend_title)
        ceg = CollectionEncoder_Generator(prepend_title, self.args, 0, 1)
        create_directory(self.args.index_root)
        create_directory(self.args.index_path)
        ceg.encode()
        self.colbert = ceg.colbert
        self.checkpoint = ceg.checkpoint 

        assert os.path.exists(self.args.index_path), self.args.index_path
        num_embeddings = sum(load_doclens(self.args.index_path))
        print("#> num_embeddings =", num_embeddings)

        import pyterrier as pt
        with pt.io.autoopen(os.path.join(self.args.index_path, "docnos.pkl.gz"), "wb") as f:
            pickle.dump(docnos, f)

        if self.args.partitions is None:
            self.args.partitions = 1 << math.ceil(math.log2(8 * math.sqrt(num_embeddings)))
            warn("You did not specify --partitions!")
            warn("Default computation chooses", self.args.partitions,
                        "partitions (for {} embeddings)".format(num_embeddings))
        index_faiss(self.args)
        print("#> Faiss encoding complete")
        endtime = timer()
        print("#> Indexing complete, Time elapsed %0.2f seconds" % (endtime - starttime))
        
