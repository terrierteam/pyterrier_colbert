import unittest
import pandas as pd
import tempfile

CHECKPOINT="http://www.dcs.gla.ac.uk/~craigm/colbert.dnn.zip"
class TestIndexing(unittest.TestCase):

    def _indexing_1doc(self, indexmgr, model, dim=None):
        #minimum test case size is 100 docs, 40 Wordpiece tokens, and nx > k. we found 200 worked
        import pyterrier as pt
        from pyterrier_colbert.indexing import ColBERTIndexer
        import os
        indexer = ColBERTIndexer(
            model, 
            os.path.dirname(self.test_dir),os.path.basename(self.test_dir), 
            chunksize=3,
            #indexmgr=indexmgr,
            gpu=False)

        if dim is not None:
          indexer.args.dim = dim
        iter = pt.get_dataset("vaswani").get_corpus_iter()
        indexer.index([ next(iter) for i in range(200) ])

        import pyterrier_colbert.pruning as pruning
        from pyterrier_colbert.ranking import ColbertPRF
            
        for factory in [indexer.ranking_factory()]:

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

    # def test_indexing_1doc_numpy(self):
    #     self._indexing_1doc('numpy')
    
    # def test_indexing_1doc_half(self):
    #     self._indexing_1doc('half')

    def test_indexing_skip_empty_docs(self):
        import pyterrier as pt
        from pyterrier_colbert.indexing import ColBERTIndexer
        import os
        indexer = ColBERTIndexer(
            model,
            os.path.dirname(self.test_dir), os.path.basename(self.test_dir),
            chunksize=3,
            # indexmgr=indexmgr,
            gpu=False,
            skip_empty_docs=True)

        if dim is not None:
            indexer.args.dim = dim
        iter = pt.get_dataset("vaswani").get_corpus_iter()
        docs = [next(iter) for i in range(200)]
        docs.insert(100, {'docno': 'empty', 'text': ''})  # truly empty
        docs.insert(105, {'docno': 'empty', 'text': ' '})  # whitespace only
        factory = indexer.index(docs)
        self.assertEqual(200, len(factory))  # check that empty docs are indeed ignored



    def indexing_docnos_correctly_empty(self):
        #A test case to see whether empty passages are handled correctly. 
        import pyterrier as pt
        from pyterrier_colbert.indexing import ColBERTIndexer
       
        import os
        indexer = ColBERTIndexer(
            CHECKPOINT, 
            os.path.dirname(self.test_dir),os.path.basename(self.test_dir), 
            chunksize=3,
            gpu=False)
        corpus = [{ "docno" : "d%d" %i, "text": "mock documents mock documents mock documentsmock documentsmock documents mock documents mock documents mock documents mock documents mock documents mock documents mock documents mock documents "+str(i) } for i in range(150)]  + [{"docno": "empty", "text":""}] +  [{ "docno" : "d%d" %(i+150), "text": "mock document of clusters (100) " + str(i) } for i in range(150)]
        with self.assertRaises(ValueError):
            indexer.index(corpus)

    def indexing_docnos_correctly_spaces(self):
        #A test case to check whether passages only containing spaces are handled correctly. 
        import pyterrier as pt
        from pyterrier_colbert.indexing import ColBERTIndexer
       
        import os
        indexer = ColBERTIndexer(
            CHECKPOINT, 
            os.path.dirname(self.test_dir),os.path.basename(self.test_dir), 
            chunksize=3,
            gpu=False)
        corpus = [{ "docno" : "d%d" %i, "text": "mock documents mock documents mock documentsmock documentsmock documents mock documents mock documents mock documents mock documents mock documents mock documents mock documents mock documents "+str(i) } for i in range(150)]  + [{"docno": "empty", "text":"  "}] +  [{ "docno" : "d%d" %(i+150), "text": "mock document of clusters (100) " + str(i) } for i in range(150)]
        with self.assertRaises(ValueError):
            indexer.index(corpus)
            
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
    
    def test_indexing_1doc_torch(self):
        self._indexing_1doc('torch', CHECKPOINT)

    def test_indexing_1doc_torch_minilm(self):
        import transformers
        if int(transformers.__version__[0]) < 4:
            self.skipTest("transfomers too old")
        from colbert.modeling.colbert import ColBERT
        model = ColBERT.from_pretrained("vespa-engine/col-minilm", query_maxlen=32, doc_maxlen=180, mask_punctuation=False, dim=32)
        self._indexing_1doc('torch', model, dim=32)

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
