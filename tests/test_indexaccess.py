import unittest
import pandas as pd
import tempfile
CHECKPOINT="http://www.dcs.gla.ac.uk/~craigm/colbert.dnn.zip"
class TestIndexing(unittest.TestCase):

    def indexing_get_embedding(self):
        #minimum test case size is 100 docs, 40 Wordpiece tokens, and nx > k. we found 200 worked
        import pyterrier as pt
        
        from pyterrier_colbert.ranking import ColBERTFactory
        factory = ColBERTFactory.from_dataset('vaswani', 'colbert_uog44k', gpu=False)
        fnt = factory.nn_term()
        df = fnt.getDF(1206)
        embs = factory.get_embeddings_by_token(1206, flatten=True)
        self.assertEqual(df, embs.shape[0])

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