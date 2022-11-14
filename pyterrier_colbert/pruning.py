import pandas as pd
import pyterrier as pt
assert pt.started(), "please run pt.init() before importing pyt_colbert"
from .ranking import ColBERTFactory

def _filter_query(query_toks_df : pd.DataFrame) -> pd.DataFrame:
    
    assert len(query_toks_df) <= 32
    
    first_row = query_toks_df.iloc[0]
    pos = query_toks_df.pos.to_numpy()
    rtr = [ first_row.qid, first_row.query, first_row.query_embs[pos],  first_row.query_toks[pos] ]    
    return pd.DataFrame([rtr], columns=["qid", "query", "query_embs", "query_toks"]) 

def _query_embedding_pruning_generic(function, cutoff : int, icf : bool = True, verbose : bool = False) -> pt.Transformer:
    return (pt.apply.by_query(function, verbose=verbose, add_ranks=True) % cutoff) >> pt.apply.by_query(_filter_query, add_ranks=False)

def query_embedding_pruning(factory : ColBERTFactory, cutoff : int, icf : bool = True, verbose : bool = False) -> pt.Transformer:
    """
    Applies the ICF or IDF based query emebedding pruning proposed in the CIKM 2021 paper.

    Arguments:
    - factory(ColBERTFactory): The ColBERTFactory object in use
    - cutoff(int): how many query emebddings to keep
    - icf(bool): to rank on collection frequency (True) or document frequency (False)
    - verbose(bool): to display a progress tqdm for this pipeline stage

    Usage::

        factory = pyterrier_colbert.ranking.ColBERTFactory(...)
        pipe = ( 
            factory.query_encoder() 
            >> pyterrier_colbert.pruning.query_embedding_pruning(factory, 9) 
            >> factory.set_retrieve(query_encoded=True)
            >> factory.index_scorer(query_encoded=False)
        )
    """
    fnt = factory.nn_term(df = not icf)    
    def make_icf_idf(query_df : pd.DataFrame) -> pd.DataFrame:        
        import math        
        assert len(query_df) == 1        
        queryrow = query_df.iloc[0]
        qid, query, embs, toks = queryrow.qid, queryrow.query, queryrow.query_embs, queryrow.query_toks
        rtr = []
        for tokenpos in range(32):
            tokenid = int(toks[tokenpos])
            ctf = fnt.getCTF_by_id(tokenid) if icf else fnt.getDF_by_id(tokenid)
            score = -ctf if ctf > 0 else -math.inf
            newrow = [qid, query, embs, tokenpos, toks, score]
            rtr.append(newrow)
        return pd.DataFrame(rtr, columns=['qid', 'query', 'query_embs', 'pos', 'query_toks', 'score'])
    return _query_embedding_pruning_generic(make_icf_idf, cutoff, verbose=verbose)
    
def query_embedding_pruning_first(factory : ColBERTFactory, cutoff : int, icf : bool = True, verbose : bool = False) -> pt.Transformer:
    """
    Applies the "First" query emebedding pruning baseline from the CIKM 2021 paper. This suppresses query embeddings based on their position in the query

    Arguments:
    - factory(ColBERTFactory): The ColBERTFactory object in use
    - cutoff(int): how many query emebddings to keep
    - icf(bool): to rank on collection frequency (True) or document frequency (False)
    - verbose(bool): to display a progress tqdm for this pipeline stage

    Usage::

        factory = pyterrier_colbert.ranking.ColBERTFactory(...)
        pipe = ( 
            factory.query_encoder() 
            >> pyterrier_colbert.pruning.query_embedding_pruning_first(factory, 9) 
            >> factory.set_retrieve(query_encoded=True)
            >> factory.index_scorer(query_encoded=False)
        )
    """
    def make_first(query_df : pd.DataFrame) -> pd.DataFrame:
        assert len(query_df) == 1
        
        queryrow = query_df.iloc[0]
        qid, query, embs, toks = queryrow.qid, queryrow.query, queryrow.query_embs, queryrow.query_toks
        rtr = []
        for tokenpos in range(32):
            newrow = [qid, query, embs, tokenpos, toks, 1/(1+tokenpos)]
            rtr.append(newrow)
        return pd.DataFrame(rtr, columns=['qid', 'query', 'query_embs', 'pos', 'query_toks', 'score'])
    return _query_embedding_pruning_generic(make_first, cutoff, verbose=verbose)

def query_embedding_pruning_special(CLS=False, Q=False, MASK=False) -> pt.Transformer:
    """
    Filters out special tokens in the ColBERT encoded query.

    NB: The tokenids are hard-coded, and hence assume a BERT model.

    Arguments:
    - CLS(bool): whether to remove the [CLS] token
    - Q(bool): whether to remove the [Q] token
    - MASK(bool): whether to remove the [MASK] tokens
    
    Usage - filter for all stages::
        factory = pyterrier_colbert.ranking.ColBERTFactory(...)
        pipe = ( 
            factory.query_encoder() 
            >> pyterrier_colbert.pruning.query_embedding_pruning_special(MASK=True) 
            >> factory.set_retrieve(query_encoded=True)
            >> factory.index_scorer(query_encoded=True)
        )

     Usage - filter for ANN stage only::
        factory = pyterrier_colbert.ranking.ColBERTFactory(...)
        pipe = ( 
            factory.query_encoder() 
            >> pyterrier_colbert.pruning.query_embedding_pruning_special(MASK=True) 
            >> factory.set_retrieve(query_encoded=True)
            >> factory.index_scorer(query_encoded=False)
        )

    """
    def row_rewriter(row):
        query_toks = row["query_toks"]
        final_mask =(query_toks > -1)
        # These tokenids are hard-coded, and hence assume a BERT model.
        Q_mask=(query_toks == 1)
        CLS_mask=(query_toks == 101)
        MASK_mask=(query_toks == 103)
        if CLS:
            final_mask = final_mask & (~CLS_mask)
        if Q:
            final_mask = final_mask & (~Q_mask)
        if MASK:
            final_mask = final_mask & (~MASK_mask)

        row["query_toks"] = query_toks[final_mask]
        row["query_embs"] = row["query_embs"][final_mask]
        return row
    return pt.apply.generic(lambda df : df.apply(row_rewriter, axis=1))