import pandas as pd
import pyterrier as pt
from pyterrier.transformer import TransformerBase
from .ranking import ColBERTFactory

def _filter_query(query_toks_df : pd.DataFrame) -> pd.DataFrame:
    
    assert len(query_toks_df) <= 32
    
    first_row = query_toks_df.iloc[0]
    pos = query_toks_df.pos.to_numpy()
    rtr = [ first_row.qid, first_row.query, first_row.query_embs[pos],  first_row.query_toks[pos] ]    
    return pd.DataFrame([rtr], columns=["qid", "query", "query_embs", "query_toks"]) 

def _query_embedding_pruning_generic(function, cutoff : int, icf : bool = True, verbose : bool = False) -> TransformerBase:
    return (pt.apply.by_query(function, verbose=verbose, add_ranks=True) % cutoff) >> pt.apply.by_query(_filter_query, add_ranks=False)

def query_embedding_pruning(factory : ColBERTFactory, cutoff : int, icf : bool = True, verbose : bool = False) -> TransformerBase:
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
    
def query_embedding_pruning_first(factory : ColBERTFactory, cutoff : int, icf : bool = True, verbose : bool = False) -> TransformerBase:
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

def query_embedding_pruning_special(CLS=False, Q=False, MASK=False) -> TransformerBase:
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

def fetch_index_encodings(factory, verbose=False, ids=False) -> TransformerBase:
    """
    New encoder that gets embeddings from rrm and stores into doc_embs column.
    If ids is True, then an additional doc_toks column is also added. This requires 
    a Faiss NN term index data structure, i.e. indexing should have ids=True set.
    input: docid, ...
    output: ditto + doc_embs [+ doc_toks]
    """
    def _get_embs(df):
        if verbose:
            import pyterrier as pt
            pt.tqdm.pandas()
            df["doc_embs"] = df.docid.progress_apply(factory.rrm.get_embedding)
        else:
            df["doc_embs"] = df.docid.apply(factory.rrm.get_embedding)
        return df

    def _get_tok_ids(df):
        fnt = factory.nn_term(False)
        def _get_toks(pid):
            end = fnt.end_offsets[pid]
            start = end - fnt.doclens[pid]
            return fnt.emb2tid[start:end]

        if verbose:
            import pyterrier as pt
            pt.tqdm.pandas()
            df["doc_toks"] = df.docid.progress_apply(_get_toks)
        else:
            df["doc_toks"] = df.docid.apply(_get_toks)
        return df
    rtr = pt.apply.by_query(_get_embs, add_ranks=False)
    if ids:
        rtr = rtr >> pt.apply.by_query(_get_tok_ids, add_ranks=False)
    return rtr

def pca_transformer(factory, pca, verbose=False) -> TransformerBase:
    """
    Apply a PCA model to the queries and documents embeddings to compress it
    input: qid, query_embs, docno, doc_embs
    output: qid, query_embs, docno, doc_embs
    """
    import torch
    def _apply_pca(df):
        iter = range(len(df))
        df["doc_embs"] = df["doc_embs"].map(lambda x : torch.from_numpy(pca.transform(x)).type(torch.float32))
        df["query_embs"] = df["query_embs"].map(lambda x : torch.from_numpy(pca.transform(x)).type(torch.float32))
        factory.args.dim = pca.n_components
        return df
    
    return pt.apply.by_query(_apply_pca, add_ranks=False)

def scorer(factory, verbose=False) -> TransformerBase:
        """
        Calculates the ColBERT max_sim operator using previous encodings of queries and documents
        input: qid, query_embs, [query_weights], docno, doc_embs
        output: ditto + score
        """
        import torch
        colbert = factory.args.colbert
        def _score_query(df):
            weightsQ = None
            Q = torch.cat([df.iloc[0].query_embs])
            if "query_weights" in df.columns:
                weightsQ = df.iloc[0].query_weights
            else:
                weightsQ = torch.ones(Q.shape[0])        
            D = torch.zeros(len(df), factory.args.doc_maxlen, factory.args.dim)
            iter = range(len(df))
            if verbose:
                iter = pt.tqdm(iter, total=len(df))
            for i in iter:
                doc_embs = df.iloc[i].doc_embs
                doclen = doc_embs.shape[0]
                D[i, 0:doclen, :] =  doc_embs
            maxscoreQ = (Q @ D.permute(0, 2, 1)).max(2).values.cpu()
            scores = (weightsQ*maxscoreQ).sum(1).cpu()
            df["score"] = scores.tolist()
            df = factory._add_docnos(df)
            return df
            
        return pt.apply.by_query(_score_query)

def blacklisted_tokens_transformer(factory, blacklist, verbose=False) -> TransformerBase:
    """
    Remove tokens and their embeddings from the document dataframe
    input: qid, query_embs, docno, doc_embs, doc_toks
    output: qid, query_embs, docno, doc_embs, doc_toks
    
    The blacklist parameters must contain a list of tokenids that should be removed
    """

    if verbose: print(f'Blacklist composed of {len(blacklist)} elements.')
    
    import numpy as np
    def _prune(row):
        
        import torch
        
        tokens = row['doc_toks']
        embeddings = row['doc_embs']
        docid = row['docid']
        qid = row['qid']
        final_mask = (tokens > -1)
        
        for element in blacklist:
            element_mask = (tokens == element)
            final_mask = final_mask & (~ element_mask)
        
        row_embs_size = embeddings.size()
        
        mask_1d = torch.cat((final_mask, torch.ones(row_embs_size[0] - final_mask.size()[0], dtype=torch.bool)))
        mask_column = torch.unsqueeze(mask_1d, 1)
        mask = mask_column.repeat(1, row_embs_size[1])
        
        row['doc_embs'] = embeddings[mask].reshape(mask_1d.count_nonzero(), row_embs_size[1])
        row['doc_toks'] = tokens[final_mask]


        pruned_embeddings = row_embs_size[0] - row['doc_embs'].size()[0]
        pruned_embeddings_percentage = pruned_embeddings/row_embs_size[0]
        if verbose:
            print(f'Embeddings removed from document {docid:10.0f}: {pruned_embeddings:10.0f} ({pruned_embeddings_percentage:10.2%})', end='\r')
        factory.add_pruning_info(qid, docid, row_embs_size[0], pruned_embeddings)
        return row

    return pt.apply.generic(lambda df : df.apply(_prune, axis=1))

import pandas as pd
import math
import json

class InfoPruning:
    '''
    Pruning Class
    '''
    
    def __init__(self):
        self.pruning_info = {}
        self.pruning_dataframes = []
        self.pruning_counter = 0

    def add_pruning_info(self, query_id, doc_id, doc_len, embeddings_pruned, topics_len=93, n_docs=10):
        self.pruning_info[query_id] = {
            doc_id: {
                'doc_len': doc_len, 
                'embeddings_pruned': embeddings_pruned
            }
        }
        self.pruning_counter += 1
        if(self.pruning_counter == topics_len * n_docs):
            self.pruning_dataframes.append(self._get_pruning_info())
            self.pruning_info = {}
            self.pruning_counter = 0
    
    def get_overall_df(self, names=[]):
        if len(names) != len(self.pruning_dataframes):
            error = f'The length of names {len(names)} must be equal to the number of dataframes {len(self.pruning_dataframes)}'
            raise ValueError(error)
        for i, element in enumerate(self.pruning_dataframes):
            element['name'] = names[i]
        print(f'Columns: {self.pruning_dataframes[0].columns}')
        final_dataframe = pd.DataFrame(columns=self.pruning_dataframes[0].columns)
        for element in self.pruning_dataframes:
            final_dataframe = final_dataframe.append(element)
        print(f'Concatenated Dataframe: {len(final_dataframe)}')
        return final_dataframe
    
    def get_reduced_df(self, names=[]):
        if len(names) != len(self.pruning_dataframes):
            error = f'The length of names {len(names)} must be equal to the number of dataframes {len(self.pruning_dataframes)}'
            raise ValueError(error)
        for i, element in enumerate(self.pruning_dataframes):
            element['name'] = names[i]
            print(element['name'])
        final_dataframe = pd.DataFrame(columns=self.pruning_dataframes[0].columns[[1, 2, 6]])
        for df in self.pruning_dataframes:
            reduced_df = df.drop(df.columns[[0, 3, 4, 5]], axis=1)
            final_dataframe = final_dataframe.append(reduced_df)
        print(f'Concatenated Dataframe: {len(final_dataframe)}')
        return final_dataframe
    
    def get_blacklist(self, factory, path, verbose=False):
        # TODO: refactor the parameter factory (maybe I can pass directly faiss_nn_term)
        faiss_nn_term = factory.nn_term(df=True)
        vocabulary = faiss_nn_term.tok.get_vocab()
        n_docs = faiss_nn_term.num_docs
        if verbose:
            print(f'Number of docs: {n_docs}')
            print(f'Vocabulary Length: {len(vocabulary)}')
        with open(path) as f:
            stopwords = json.load(f)
        if verbose: print("Stopwords length:", len(stopwords))
        blacklist_tids = []

        for stopword in stopwords:
            if stopword in vocabulary:
                blacklist_tids.append(vocabulary[stopword])

        # Remove items with 0 document frequency
        if verbose: print("Blacklist length:", len(blacklist_tids))
        blacklist_tids_dfs = []
        for tid in blacklist_tids:
            df = factory.nn_term(df=True).getDF_by_id(tid)
            idf = math.log(n_docs/(df + 1), 10)
            if df != 0: blacklist_tids_dfs.append((tid, idf))
        if verbose: print("Blacklist length (without 0 df elements):", len(blacklist_tids_dfs))
        # order by inverse document frequency
        ordered_blacklist = sorted(blacklist_tids_dfs, key= lambda pair: pair[1])
        final_blacklist = []
        for _id, _ in ordered_blacklist: final_blacklist.append(_id)
        return final_blacklist
            
    def _get_pruning_info(self):
        rows = []
        for query_id, query_data in self.pruning_info.items():
            row = self._get_pruning_info_per_query_data(query_id, query_data)
            rows.append(row)
        df = pd.DataFrame(data=rows,
            columns=['qid', '# total embeddings', '# tokens pruned', 'tokens pruned %', 'most pruned document', 'less pruned document'])
        return df
            
    def _get_pruning_info_per_query_data(self, query_id, query_data):
        total_embeddings = 0
        total_prunings = 0
        pruning_percentages = []
        for key, value in query_data.items():
            total_embeddings += value['doc_len']
            total_prunings += value['embeddings_pruned']
            pruning_percentages.append((key, value['embeddings_pruned']/value['doc_len']))
        overall_percentage = round(total_prunings/total_embeddings * 100, 2)
        max_pruned = max(pruning_percentages, key= lambda t: t[1])
        min_pruned = min(pruning_percentages, key= lambda t: t[1])
        max_pruned_str = f'{max_pruned[0]:4} ({max_pruned[1]:4.2%})'
        min_pruned_str = f'{min_pruned[0]:4} ({min_pruned[1]:4.2%})'
        return [query_id, total_embeddings, total_prunings, overall_percentage, max_pruned_str, min_pruned_str]