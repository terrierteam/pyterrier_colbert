import unittest
import tempfile

CHECKPOINT="http://www.dcs.gla.ac.uk/~craigm/colbert.dnn.zip"
class TestMaskPunctuation(unittest.TestCase):

    def test_mask_punctuation_tokenisation(self):
        import pyterrier as pt
        from pyterrier_colbert.ranking import ColBERTModelOnlyFactory
        mof = ColBERTModelOnlyFactory(CHECKPOINT, gpu=False, mask_punctuation=True)
        inference = mof.args.inference
        self.assertTrue(len(inference.colbert.skiplist) > 0)
        embs, ids = inference.docFromText(["bottom. --Lynne @ Lynne Marie Studios, Inc."], bsize=8, with_ids=True, keep_dims=False)
        # [(3953, 'bottom'),
        # (1012, '.'), REMOVED
        # (1011, '-'), REMOVED
        # (1011, '-'), REMOVED
        # (26938, 'lynne'),
        # (1030, '@'), REMOVED
        # (26938, 'lynne'),
        # (5032, 'marie'),
        # (4835, 'studios'),
        # (1010, ','), REMOVED
        # (4297, 'inc'),
        # (1012, '.')] REMOVED
        # 6 KEPT
        expected_ids = [101, 2] + [3953, 26938, 26938, 5032, 4835, 4297] + [102]
        self.assertEqual(len(ids[0].tolist()), len(expected_ids))
        self.assertTrue(ids[0].tolist() == expected_ids)
        self.assertEqual(embs[0].shape[0], len(expected_ids))
    
    def test_nomask_punctuation_tokenisation(self):
        import pyterrier as pt
        from pyterrier_colbert.ranking import ColBERTModelOnlyFactory
        mof = ColBERTModelOnlyFactory(CHECKPOINT, gpu=False, mask_punctuation=False)
        inference = mof.args.inference
        self.assertTrue(len(inference.colbert.skiplist) == 0)
        embs, ids = inference.docFromText(["bottom. --Lynne @ Lynne Marie Studios, Inc."], bsize=8, with_ids=True, keep_dims=False)
        # [(3953, 'bottom'),
        # (1012, '.'),
        # (1011, '-'),
        # (1011, '-'),
        # (26938, 'lynne'),
        # (1030, '@'),
        # (26938, 'lynne'),
        # (5032, 'marie'),
        # (4835, 'studios'),
        # (1010, ','),
        # (4297, 'inc'),
        # (1012, '.')]
        expected_ids = [101, 2] + [3953, 1012, 1011, 1011, 26938, 1030, 26938, 5032, 4835, 1010, 4297, 1012] + [102]
        self.assertEqual(len(ids[0].tolist()), len(expected_ids))
        self.assertTrue(ids[0].tolist() == expected_ids)
        self.assertEqual(embs[0].shape[0], len(expected_ids))


    def test_mask_punctuation_scoring(self):
        import pyterrier as pt
        from pyterrier_colbert.indexing import ColBERTIndexer

        index_root = self.test_dir

        iter = pt.get_dataset("vaswani").get_corpus_iter()
        to_index = [next(iter) for i in range(200) ] + [{"docno": "a", "text": "bottom. --Lynne @ Lynne Marie Studios, Inc."}]
        query = "Can anyone tell me the dimensions of this pot? Is clear top glass? ASAP, please?"

        # indexing without ids and with masking
        indexer = ColBERTIndexer(
            CHECKPOINT, 
            index_root, "no_ids", 
            chunksize=3,
            gpu=False,
            mask_punctuation=True,
            ids=False)
        indexer.index(to_index)
        pytcolbert = indexer.ranking_factory()
        dense_e2e = pytcolbert.end_to_end()

        no_ids = dense_e2e.search(query)
        no_ids = no_ids[no_ids["docno"] == "a"].iloc[0]

        # indexing with ids and with masking
        indexer = ColBERTIndexer(
            CHECKPOINT, 
            index_root, "with_ids", 
            chunksize=3,
            gpu=False,
            mask_punctuation=True,
            ids=True)
        indexer.index(to_index)
        pytcolbert = indexer.ranking_factory()
        dense_e2e = pytcolbert.end_to_end()

        with_ids = dense_e2e.search(query)
        with_ids = with_ids[with_ids["docno"] == "a"].iloc[0]

        assert with_ids["score"] == no_ids["score"]

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
