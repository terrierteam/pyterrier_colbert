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
        from pyterrier_colbert.ranking import ColBERTModelOnlyFactory
        self.factory = ColBERTModelOnlyFactory(checkpoint, gpu=False)
        self.df = pt.new.ranked_documents([[1, 2]])
        self.df["text"] = [ "professor proton mixed the chemicals", "chemical brothers played that tune"]
        self.df["query"] = ["chemical reactions", "chemical reactions"]

    def test_text_scorer(self):
        scorer = self.factory.text_scorer()
        rtr = scorer.transform(self.df)
        self.assertTrue("score" in rtr.columns)

    def test_query_encoder(self):
        queries = pt.new.queries(["chemical reactions"])
        qend = self.factory.query_encoder()
        rtr = qend(queries)
        self.assertTrue("query_embs" in rtr.columns)
        self.assertTrue("query_toks" in rtr.columns)

    def test_text_scorer_with_qembs(self):
        scorer = self.factory.text_scorer()
        qe_scorer =  pt.transformer.SourceTransformer(self.df) >> self.factory.query_encoder() >> self.factory.text_scorer(query_encoded=True)  
        rtr = scorer.transform(self.df)  
        qe_rtr = qe_scorer.search(self.df["query"])
        self.assertTrue("score" in rtr.columns)
        self.assertTrue(rtr.equals(qe_rtr))
