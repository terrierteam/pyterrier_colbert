import unittest
import pandas as pd
import tempfile
class TestApprox(unittest.TestCase):

    def setUp(self):
        import pyterrier as pt
        if not pt.started():
            pt.init()

    def _checkimpl(self, fn):
        import numpy as np
        mapping = np.array([ 0, 0, 1, 1, 1 ])
        faiss_ids = np.array([[0,1,2], [0, 2, 4]])
        #mapped to docids:
        #[0,0,1] [0, 1, 1]
        faiss_scores = np.array([[2.1, 2.0, 1.9], [1.3, 1.2, 1.1]])
        weights = np.array([4, 5])
        num_docs = 2
        score_buffer = np.zeros((num_docs, faiss_ids.shape[0] ))

        rtr = fn(faiss_scores, faiss_ids, mapping, weights, score_buffer)
        assert rtr[1][0] == (weights[0] * 2.1 + weights[1] * 1.3)
        assert rtr[1][1] == (weights[0] * 1.9 + weights[1] * 1.2), (weights[0] * 1.9 + weights[1] * 1.2)

    def test_maxsim_np(self):
        import pyterrier_colbert.ranking
        self._checkimpl(pyterrier_colbert.ranking._approx_maxsim_numpy)

    def test_maxsim_defaultdict(self):
        import pyterrier_colbert.ranking
        self._checkimpl(pyterrier_colbert.ranking._approx_maxsim_defaultdict)


    


