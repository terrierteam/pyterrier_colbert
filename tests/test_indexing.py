import unittest
import pandas as pd
import tempfile
class TestIndexing(unittest.TestCase):

    def test_indexing_1doc(self):
        #minimum test case size is 100 docs, 40 Wordpiece tokens, and nx > k. we found 200 worked
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
        indexer.index([ next(iter) for i in range(200) ])
            
        for factory in [indexer.ranking_factory()]:

            dfOut = factory.end_to_end().search("chemical reactions")
            self.assertTrue(len(dfOut) > 0)

            dfOut = factory.prf(False).search("chemical reactions")
            self.assertTrue(len(dfOut) > 0)

            dfOut = factory.prf(True).search("chemical reactions")
            self.assertTrue(len(dfOut) > 0)


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