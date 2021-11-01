import unittest
import tempfile
from unittest.result import failfast
import pyterrier as pt
class TestIndexing(unittest.TestCase):

    def setUp(self):
        from pyterrier_colbert.ranking import ColBERTFactory
        if not pt.started():
            pt.init()
        self.test_dir = tempfile.mkdtemp()
        checkpoint="http://www.dcs.gla.ac.uk/~craigm/colbert.dnn.zip"
        self.factory = ColBERTFactory(checkpoint, None, None, gpu=False)

    def test_text_scorer(self):
        scorer = self.factory.text_scorer()
        df = pt.new.ranked_documents([[1, 2]])
        df["text"] = [ "professor proton mixed the chemicals", "chemical brothers played that tune"]
        df["query"] = ["chemical reactions", "chemical reactions"]
        rtr = scorer.transform(df)
        self.assertTrue("score" in rtr.columns)

    def test_text_scorer_with_qembs(self):
        df = pt.new.ranked_documents([[1, 2]])
        df["text"] = [ "professor proton mixed the chemicals", "chemical brothers played that tune"]
        df["query"] = ["chemical reactions", "chemical reactions"]
        scorer = self.factory.text_scorer()
        qe_scorer =  pt.transformer.SourceTransformer(df) >> self.factory.query_encoder() >> self.factory.text_scorer(query_encoded=True)  
        rtr = scorer.transform(df)  
        qe_rtr = qe_scorer.search(df["query"])
        self.assertTrue("score" in rtr.columns)
        self.assertTrue(rtr.equals(qe_rtr))
