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