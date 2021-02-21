# pyterrier-colbert

Advanced [PyTerrier](https://github.com/terrier-org/pyterrier) bindings for [ColBERT](https://github.com/stanford-futuredata/ColBERT/tree/v0.2), including for dense indexing and retrieval.


## Usage

Given an existing ColBERT checkpoint, an end-to-end ColBERT dense retrieval index can be created as follows:

```python
from pyterrier_colbert.indexing import ColBERTIndexer
indexer = ColBERTIndexer("/path/to/checkpoint.dnn", "/path/to/index", "index_name")
indexer.index(dataset.get_corpus_iter())
```

An end-to-end ColBERT dense retrieval pipeline can be formulated as follows:

```python
from pyterrier_colbert.ranking import ColBERTFactory
pytcolbert = ColBERTFactory("/path/to/checkpoint.dnn", "/path/to/index", "index_name")
dense_e2e = pytcolbert.set_retrieve() >> pytcolbert.index_scorer()
```

A ColBERT re-ranker of BM25 can be formulated as follows (you will need to have the [text saved in your Terrier index](https://pyterrier.readthedocs.io/en/latest/text.html)):

```python
bm25 = pt.BatchRetrieve(terrier_index, wmodel="BM25", metadata=["docno", "text"])
sparse_colbert = bm25 >> pytcolbert.text_scorer()
```

Thereafter it is easy to conduct a side-by-side comparison of effectiveness:

```python
pt.Experiment(
    [sparse_colbert, dense_e2e]
    dataset.get_topics(),
    dataset.get_qrels(),
    measures=["map", "ndcg_cut_10"],
    names=["BM25 >> ColBERT", "Dense ColBERT"]
)
```


## References

 - [Khattab20]: Omar Khattab, Matei Zaharia. ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT. In Proceedings of SIGIR 2020. https://arxiv.org/abs/2004.12832
 - [Macdonald20]: Craig Macdonald, Nicola Tonellotto. Declarative Experimentation inInformation Retrieval using PyTerrier. Craig Macdonald and Nicola Tonellotto. In Proceedings of ICTIR 2020. https://arxiv.org/abs/2007.14271


## Credits

 - Craig Macdonald, University of Glasgow
 - Nicola Tonellotto, University of Pisa

