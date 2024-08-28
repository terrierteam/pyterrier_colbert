# pyterrier-colbert & ColBERT-PRF

Advanced [PyTerrier](https://github.com/terrier-org/pyterrier) bindings for [ColBERT](https://github.com/stanford-futuredata/ColBERT/), including for dense indexing and retrieval. This also includes the implementations of [ColBERT PRF](https://arxiv.org/abs/2106.11251), [approximate ANN scoring](https://arxiv.org/abs/2108.11480) and [query embedding pruning](https://arxiv.org/abs/2108.10341). 

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

A ColBERT re-ranker of BM25 can be formulated as follows (you will need to have an index with [text saved](https://pyterrier.readthedocs.io/en/latest/text.html) - the Terrier data repostiory conviniently [provides such an index](http://data.terrier.org/msmarco_passage.dataset.html#terrier_stemmed_text)):
```python
bm25 = pt.terrier.Retriever.from_dataset('msmarco_passage', 'terrier_stemmed_text', wmodel='BM25', metadata=['docno', 'text'])
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

## Approximate ANN Scoring and Query Embedding Pruning

This repository contains code to apply the techniques of query embedding pruning [Tonellotto21] and approximate ANN ranking [Macdonald21a].

Query Emebdding pruning can be applied using the following pipeline:
```python
qep_pipe5 = (factory.query_encoder() 
            >> pyterrier_colbert.pruning.query_embedding_pruning(factory, 5) 
            >> factory.set_retrieve(query_encoded=True)
            >> factory.index_scorer(query_encoded=False)
)
```
where 5 is the number of query embeddings based on collection frquency to retain.

Approximate ANN scoring can be applied using the following pipeline:
```python
ann_pipe = (factory.ann_retrieve_score() % 200) >> factory.index_scorer(query_encoded=True)
```
where 200 is the number of top-scored ANN candidates to forward for exact scoring.


## Demos
 - vaswani.ipynb - [[Github](vaswani.ipynb)] [[Colab](https://colab.research.google.com/github/cmacdonald/pyterrier_colbert/blob/main/vaswani.ipynb)] - demonstrates end-to-end dense retrieval and indexing on the Vaswani corpus (~11k documents)
 - colbertprf-msmarco-passages.ipynb - [[Github](colbertprf-msmarco-passages.ipynb)] - demonstrates ColBERT PRF on the TREC Deep Learning track (MSMARCO) passage ranking tasks.
 - cikm2021-demos.ipynb - [[Github](cikm2021-demos.ipynb)] - demonstrates ANN scoring and Query Embedding Pruning on the TREC Deep Learning track (MSMARCO) passage ranking tasks.
 - colbert_text_and_explain.ipynb - [[Github](colbert_text_and_explain.ipynb)] [[Colab](https://colab.research.google.com/github/cmacdonald/pyterrier_colbert/blob/main/colbert_text_and_explain.ipynb)] - demonstrates using a ColBERT model for scoring text, and for explaining an interaction. If you use one of these interaction diagrams, please cite [Macdonald21].

## Resource Requirements

You will need a GPU to use this. Preferable more than one. You will also need lots of RAM - ColBERT requires you load the entire index into memory.

| Name               | Corpus size   | Indexing Time         | Index Size |
| -------------------| ------------- | --------------------- | ---------- |
| Vaswani            | 11k abstracts | 2 minutes (1 GPU)     | 163 MB     |
| MSMARCO Passages   | 8M passages   | ~24 hours (1 GPU)     | 192 GB     |

## Installation

This package can be installed using Pip, and then used with PyTerrier. See also the examples notebooks.

```shell
pip install -q git+https://github.com/terrierteam/pyterrier_colbert.git
conda install -c pytorch faiss-gpu=1.6.5 # or faiss-cpu
#on Colab: pip install faiss-gpu==1.6.5
```

NB: ColBERT requires FAISS, namely the faiss-gpu package, to be installed. `pip install faiss-gpu` does **NOT** usually work.
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
 - Muhammad Hammad Khan, University of Glasgow
 - Sean MacAvaney, University of Glasgow
 - Sasha Petrov, University of Glasgow
