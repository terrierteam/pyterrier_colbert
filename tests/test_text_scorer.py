import unittest
import tempfile
from unittest.result import failfast
import pyterrier as pt

class TestIndexing(unittest.TestCase):

    def setUp(self):
        import pyterrier as pt
        if not pt.started():
            pt.init()
        self.test_dir = tempfile.mkdtemp()

    def test_text_scorer(self):
        from pyterrier_colbert.ranking import ColBERTFactory
        checkpoint="http://www.dcs.gla.ac.uk/~craigm/colbert.dnn.zip"
        factory = ColBERTFactory(checkpoint, None, None, gpu=False)
        scorer = factory.text_scorer()
        df = pt.new.ranked_documents([[1, 2]])
        df["text"] = [ "professor proton mixed the chemicals", "chemical brothers played that tune"]
        df["query"] = ["chemical reactions", "chemical reactions"]
        rtr = scorer.transform(df)
        self.assertTrue("score" in rtr.columns)

