# STAR

**Structure and Text-Aware Relational meta-learning for few-shot molecular property prediction**

STAR uses assay text descriptions and molecular structure information for few-shot molecular property prediction. By default, the code enables all three STAR modules:

- Semantic task sampler
- L-TAFM task-adaptive feature modulation
- SK-Rel structure-knowledge relation module

<p align="center">
  <img src="figure/overview.jpg" width="85%" alt="STAR overview">
</p>

## Environment

Main dependencies:

```text
python 3.7
torch 1.7.1
torch_geometric 1.6.3
torch_scatter 2.0.7
learn2learn 0.1.6
rdkit 2022.9.3
scikit-learn
tensorboard
```

## Data

Put benchmark data under:

```text
data/
```

The `data/` directory is ignored by git because the datasets are large.

## Generate Text Embeddings

Before running STAR, generate assay/property text embeddings:

```bash
python scripts/precompute_property_embeddings.py --dataset tox21 --encoder tfidf
```

For all supported datasets:

```bash
python scripts/precompute_property_embeddings.py --dataset all --encoder tfidf
```

## Run

```bash
python run.py --dataset tox21 --n_support 10 --gpu 0
```
