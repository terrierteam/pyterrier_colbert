# pyterrier-colbert & ColBERT-PRF

Advanced [PyTerrier](https://github.com/terrier-org/pyterrier) bindings for [ColBERT](https://github.com/stanford-futuredata/ColBERT/), including for dense indexing and retrieval. This also includes the implementation of [ColBERT PRF](https://arxiv.org/abs/2106.11251). 

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
dense_e2e = pytcolbert.end_to_end()
```

A ColBERT re-ranker of BM25 can be formulated as follows (you will need to have the [text saved in your Terrier index](https://pyterrier.readthedocs.io/en/latest/text.html)):

```python
bm25 = pt.BatchRetrieve(terrier_index, wmodel="BM25", metadata=["docno", "text"])
sparse_colbert = bm25 >> pytcolbert.text_scorer()
```

Thereafter it is possible to conduct a side-by-side comparison of effectiveness:

```python
pt.Experiment(
    [bm25, sparse_colbert, dense_e2e],
    dataset.get_topics(),
    dataset.get_qrels(),
    eval_metrics=["map", "ndcg_cut_10"],
    names=["BM25", "BM25 >> ColBERT", "Dense ColBERT"]
)
```

## ColBERT PRF

You can use ColBERTFactory to obtain [ColBERT PRF](https://arxiv.org/abs/2106.11251) pipelines, as follows:
```python
colbert_prf_rank = pytcolbert.prf(rerank=False)
colbert_prf_rerank = pytcolbert.prf(rerank=True)
```

ColBERT PRF requires the ColBERT index to have aligned token ids. During indexing, use the `ids=True` kwarg for ColBERTIndexer, as follows:
```python
indexer = ColBERTIndexer("/path/to/checkpoint.dnn", "/path/to/index", "index_name", ids=True)
```

If you use ColBERT PRF in your research, you must cite our ICTIR 2021 paper (citation included below).

All of our results files are available from the paper's [Virtual Appendix](https://github.com/Xiao0728/ColBERT-PRF-VirtualAppendix).

## Coming Soon

This repository will shorly be updated with code to apply the techniques of query embedding pruning [Tonellotto21] and approximate ANN ranking [Macdonald21a].

## Demos
 - vaswani.ipy - [[Github](vaswani.ipynb)] [[Colab](https://colab.research.google.com/github/cmacdonald/pyterrier_colbert/blob/main/vaswani.ipynb)] - demonstrates end-to-end dense retrieval and indexing on the Vaswani corpus (~11k documents)
 - colbertprf-msmarco-passages.ipynb - [[Github](colbertprf-msmarco-passages.ipynb)] - demonstrates ColBERT PRF on the TREC Deep Learning track (MSMARCO) passage ranking tasks.
 - colbert_text_and_explain.ipynb - [[Github](colbert_text_and_explain.ipynb)] [[Colab](https://colab.research.google.com/github/cmacdonald/pyterrier_colbert/blob/main/colbert_text_and_explain.ipynb)] - demonstrates using a ColBERT model for scoring text, and for explaining an interaction. If you use one of these interaction diagrams, please cite [Macdonald21].

## Resource Requirements

You will need a GPU to use this. Preferable more than one. You will also need lots of RAM - ColBERT requires you load the entire index into memory.

| Name               | Corpus size   | Indexing Time         | Index Size |
| -------------------| ------------- | --------------------- | ---------- |
| Vaswani            | 11k abstracts | 2 minutes (1 GPU)     | 163 MB     |
| MSMARCO Passages   | 8M passages   | ~24 hours (1 GPU)     | 192 GB     |

## Installation

ColBERT requires FAISS, namely the faiss-gpu package, to be installed. `pip install faiss-gpu` does **NOT** usually work.
FAISS [recommends using Anaconda](https://github.com/facebookresearch/faiss/blob/master/INSTALL.md) to install faiss-gpu.
On Colab, you need to resort to pip install. We recommend faiss-gpu version 1.6.3, not 1.7.0.


## References

 - [Khattab20]: Omar Khattab, Matei Zaharia. ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT. In Proceedings of SIGIR 2020. https://arxiv.org/abs/2004.12832
 - [Macdonald20]: Craig Macdonald, Nicola Tonellotto. Declarative Experimentation in Information Retrieval using PyTerrier. Craig Macdonald and Nicola Tonellotto. In Proceedings of ICTIR 2020. https://arxiv.org/abs/2007.14271
 - [Macdonald21]: On Single and Multiple Representations in Dense Passage Retrieval. Craig Macdonald, Nicola Tonellotto and Iadh Ounis. In Proceedings of IIR 2021. https://arxiv.org/abs/2108.06279
 - [Macdonald21a]: On Approximate Nearest Neighbour Selection for Multi-Stage Dense Retrieval. Craig Macdonald and Nicola Tonellotto. In Proceedings of CIKM 2021. https://arxiv.org/abs/2108.11480 
 - [Tonellotto21]: Query Embedding Pruning for Dense Retrieval Nicola Tonellotto and Craig Macdonald. In Proceedings of CIKM 2021. https://arxiv.org/abs/2108.10341
 - [Wang21]: Xiao Wang, Craig Macdonald, Nicola Tonellotto, Iadh Ounis. Pseudo-Relevance Feedback for Multiple Representation Dense Retrieval. In Proceedings of ICTIR 2021. https://arxiv.org/abs/2106.11251


## Credits

 - Craig Macdonald, University of Glasgow
 - Nicola Tonellotto, University of Pisa
 - Sanjana Karumuri, University of Glasgow
 - Xiao Wang, University of Glasgow
