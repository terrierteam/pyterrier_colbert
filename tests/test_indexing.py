import unittest
import pandas as pd
import tempfile

CHECKPOINT="http://www.dcs.gla.ac.uk/~craigm/colbert.dnn.zip"
class TestIndexing(unittest.TestCase):

    def _indexing_1doc(self, indexmgr, indexread):
        #minimum test case size is 100 docs, 40 Wordpiece tokens, and nx > k. we found 200 worked
        import pyterrier as pt
        from pyterrier_colbert.indexing import ColBERTIndexer
        import os
        indexer = ColBERTIndexer(
            CHECKPOINT, 
            os.path.dirname(self.test_dir),os.path.basename(self.test_dir), 
            chunksize=3,
            indexmgr=indexmgr,
            gpu=False)

        iter = pt.get_dataset("vaswani").get_corpus_iter()
        indexer.index([ next(iter) for i in range(200) ])

        import pyterrier_colbert.pruning as pruning
        from pyterrier_colbert.ranking import ColbertPRF
            
        for factory in [indexer.ranking_factory()]:
            factory.memtype = indexread

            self.assertEqual(200, len(factory))

            for pipe, has_score, name in [
                (factory.end_to_end(), True, "E2E"),
                (factory.prf(False), True, "PRF rank"),
                (factory.prf(True), True, "PRF rerank"),
                (factory.set_retrieve(), False, "set_retrieve"),
                (factory.ann_retrieve_score() , True, "approx"),
                ((
                    factory.query_encoder() 
                    >> pruning.query_embedding_pruning_first(factory, 8) 
                    >> factory.set_retrieve(query_encoded=True)
                    >> factory.index_scorer(query_encoded=False) 
                    ), True, "QEP first"),
                ((
                    factory.query_encoder() 
                    >> pruning.query_embedding_pruning(factory, 8) 
                    >> factory.set_retrieve(query_encoded=True)
                    >> factory.index_scorer(query_encoded=False) 
                    ), True, "QEP ICF"),
                ((
                    factory.query_encoder() 
                    >> pruning.query_embedding_pruning_special(CLS=True) 
                    >> factory.set_retrieve(query_encoded=True)
                    >> factory.index_scorer(query_encoded=False) 
                    ), True, "QEP CLS"),
                ((
                    factory.query_encoder() >> factory.ann_retrieve_score(query_encoded=True)
                    ), True, "ANN with query encoded"),
                ((
                    factory.query_encoder() 
                    >> factory.ann_retrieve_score(query_encoded=True)
                    >> ColbertPRF(factory, fb_docs=3, fb_embs=10, beta=1.0, k=24, return_docs=True)
                    >> factory.index_scorer(query_encoded=True) 
                    ), True, "PRF rerank and ANN with query encoded"),
                ((
                    factory.query_encoder() 
                    >> factory.ann_retrieve_score(query_encoded=True)
                    >> ColbertPRF(factory, fb_docs=3, fb_embs=10, beta=1.0, k=24, return_docs=False)
                    >> factory.ann_retrieve_score(query_encoded=True)
                    >> factory.index_scorer(query_encoded=True) 
                    ), True, "PRF rank and ANN with query encoded"),
            ]:
                with self.subTest(name):
                    print("Running subtest %s" % name)
                    dfOut = pipe.search("chemical reactions")                
                    self.assertTrue(len(dfOut) > 0)
                    
                    if has_score:
                        self.assertTrue("score" in dfOut.columns)
                    else:
                        self.assertFalse("score" in dfOut.columns)

    def indexing_empty(self):
        #minimum test case size is 100 docs, 40 Wordpiece tokens, and nx > k. we found 200 worked
        import pyterrier as pt
        from pyterrier_colbert.indexing import ColBERTIndexer
        checkpoint="http://www.dcs.gla.ac.uk/~craigm/colbert.dnn.zip"
        import os
        indexer = ColBERTIndexer(
            CHECKPOINT, 
            os.path.dirname(self.test_dir),os.path.basename(self.test_dir), 
            chunksize=3,
            gpu=False)

        iter = pt.get_dataset("vaswani").get_corpus_iter()
        indexer.index([ next(iter) for i in range(200) ] +  [{"docno": "a", "text": ""}])

    def indexing_merged(self):
        #minimum test case size is 100 docs, 40 Wordpiece tokens, and nx > k. we found 200 worked
        import pyterrier as pt
        from pyterrier_colbert.indexing import ColBERTIndexer, merge_indices
        checkpoint="http://www.dcs.gla.ac.uk/~craigm/colbert.dnn.zip"
        import os
        index_root = self.test_dir
        indexer = ColBERTIndexer(
            CHECKPOINT, 
            index_root, "index_part_0", 
            chunksize=3,
            gpu=False)

        iter = pt.get_dataset("vaswani").get_corpus_iter()
        indexer.index([ next(iter) for i in range(200) ])
        indexer = ColBERTIndexer(
            CHECKPOINT, 
            index_root, "index_part_1", 
            chunksize=3,
            gpu=False)

        iter = pt.get_dataset("vaswani").get_corpus_iter()
        indexer.index([ next(iter) for i in range(200) ])
        
        merge_indices(index_root, "index_part")
        from pyterrier_colbert.ranking import ColBERTFactory
        factory = ColBERTFactory(CHECKPOINT, index_root, "index_part", faiss_partitions=100, gpu=False)
        self.assertEqual(400, len(factory.docid2docno))
    
    def test_indexing_1doc_torch_mem(self):
        self._indexing_1doc('torch', "mem")

    # def test_indexing_1doc_torch_mem(self):
    #     self._indexing_1doc('half', "mmap")

    def test_indexing_1doc_numpy_mem(self):
        self._indexing_1doc('numpy', 'numpy')
    
    def test_indexing_1doc_numpy_mmap(self):
        self._indexing_1doc('numpy', 'numpy_mmap')

    def setUp(self):
        import pyterrier as pt
        if not pt.started():
            pt.init()
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        try:
            shutil.rmtree(self.test_dir)
        except:
            pass