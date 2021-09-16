import unittest
import pandas as pd
import tempfile
class TestIndexing(unittest.TestCase):

    def test_indexing_1doc(self):
        #minimum test case size is 100 docs, 40 Wordpiece tokens
        import pyterrier as pt
        from pyterrier_colbert.indexing import ColBERTIndexer
        checkpoint="http://www.dcs.gla.ac.uk/~craigm/colbert.dnn.zip"
        import os
        indexer = ColBERTIndexer(
            checkpoint, 
            os.path.dirname(self.test_dir),os.path.basename(self.test_dir), 
            chunksize=3,
            gpu=False)

        iter = pt.get_dataset("vaswani").get_corpus_iter()
        indexer.index([ next(iter) for i in range(100) ])
            
        factory = indexer.ranking_factory()


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