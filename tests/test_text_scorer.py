import unittest
import tempfile
from unittest.result import failfast
import pyterrier as pt

class TestTextScoring(unittest.TestCase):

    def setUp(self):
        if not pt.started():
            pt.init()
        self.test_dir = tempfile.mkdtemp()
        checkpoint="http://www.dcs.gla.ac.uk/~craigm/colbert.dnn.zip"
        from pyterrier_colbert.ranking import ColBERTModelOnlyFactory
        self.factory = ColBERTModelOnlyFactory(checkpoint, gpu=False)
        self.df = pt.new.ranked_documents([[1, 2]])
        self.df["text"] = [ "professor proton mixed the chemicals", "chemical brothers played that tune"]
        self.df["query"] = ["chemical reactions", "chemical reactions"]

    def test_prf_text(self):
        from pyterrier_colbert.ranking import ColbertPRF

        basescorer = self.factory.text_scorer()
        basertr = basescorer.transform(self.df).sort_values('docno')

        # monkey patch in an FNT from another index 
        from pyterrier_colbert.ranking import ColBERTFactory
        basefactory = ColBERTFactory.from_dataset('vaswani', 'colbert_uog44k', gpu=False)
        self.factory.nn_term = basefactory.nn_term
        pipe = (
            self.factory.query_encoder() 
            >> self.factory.text_encoder() 
            >> ColbertPRF(self.factory, 5, 2, return_docs=True, fb_docs=2)
            >> self.factory.scorer(gpu=False)
        )
        
        prfrtr = pipe.transform(self.df).sort_values('docno')
        self.assertGreater(prfrtr.iloc[0].score, basertr.iloc[0].score)
        self.assertGreater(prfrtr.iloc[1].score, basertr.iloc[1].score)
        

    def test_text_encoder(self):
        enc = self.factory.text_encoder()
        rtr1 = enc.transform(self.df)
        self.assertTrue("doc_embs" in rtr1.columns)
        self.assertTrue("doc_toks" in rtr1.columns)
        t1 = rtr1.iloc[0].doc_embs
        self.assertEqual(128, t1.shape[1])
        t2 = rtr1.iloc[1].doc_embs
        self.assertEqual(128, t2.shape[1])

    def test_text_scorer_cmp(self):
        scorer1 = self.factory.text_scorer()
        rtr1 = scorer1.transform(self.df).sort_values('docno')
        self.assertTrue("score" in rtr1.columns)

        scorer2 = self.factory.query_encoder() >> self.factory.text_encoder() >> self.factory.scorer(gpu=False)
        rtr2 = scorer2.transform(self.df).sort_values('docno')
        self.assertTrue("score" in rtr2.columns)

        self.assertEqual(rtr1.iloc[0].docno, rtr2.iloc[0].docno)
        self.assertAlmostEqual(rtr1.iloc[0].score, rtr2.iloc[0].score, 5)
        self.assertAlmostEqual(rtr1.iloc[1].score, rtr2.iloc[1].score, 5)

    def test_query_encoder(self):
        queries = pt.new.queries(["chemical reactions"])
        qend = self.factory.query_encoder()
        rtr = qend(queries)
        self.assertTrue("query_embs" in rtr.columns)
        self.assertTrue("query_toks" in rtr.columns)

    def test_text_scorer_with_qembs(self):
        scorer = self.factory.text_scorer()
        qe_scorer = pt.transformer.SourceTransformer(self.df) >> self.factory.query_encoder() >> self.factory.text_scorer(query_encoded=True)  
        rtr = scorer.transform(self.df)  
        qe_rtr = qe_scorer.search(self.df["query"])
        self.assertTrue("score" in rtr.columns)
        self.assertTrue(rtr.equals(qe_rtr))
