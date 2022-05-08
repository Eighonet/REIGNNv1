# REIGNN | Version 1 

![Pipeline_image](model/images/recommender_pipeline_rev4.png#gh-light-mode-only)
![Pipeline_image](model/images/recommender_pipeline_rev4dm.png#gh-dark-mode-only)

Welcome to the official repo of the REIGNNv1 model -- GNN-based recommender system for scientific collaborations assessment. Here we present the source code for ISWC'22 paper "Recommendations Become Even More Useful:
Multi-task Extension of Scientific Collaborations Forecasting".

Vladislav Tishin, Artyom Sosedka, Natalia Semenova, Anastasia Martynova, [Vadim Porvatov](https://www.researchgate.net/profile/Vadim-Porvatov)

PDF: _to be added_.

# Prerequisites

```
numpy==1.19.5
pandas==1.3.0
torch==1.8.1
torch-sparse==0.6.12  
torch-scatter==2.0.8
torch-cluster==1.5.9
torch-spline-conv==1.2.1
torch-geometric==2.0.3
wandb==0.12.9
```

# Datasets

For the evaluation purposes, we established two datasets of different size which could be used for evaluation of the REIGNN model. The initial data was gathered from the [Semantic Scholar Open Research Corpus](https://api.semanticscholar.org/corpus) and [SCImago Journal & Country Rank
website](https://www.scimagojr.com).

<table>
  <tr>
    <td>Properties</td>
    <td colspan="2">CS1021<sub>small</sub></td>
    <td colspan="2">CS1021<sub>medium</sub></td>
  </tr>
  <tr>
    <td>Network type</td>
    <td>Citation</td>
    <td>Co-authorship</td>
    <td>Citation</td>
    <td>Co-authorship</td>
  </tr>
  
  <tr>
    <td>Nodes</td>
    <td>3164</td>
    <td>1412</td>
    <td>53916</td>
    <td>8432</td>
  </tr>
  
  <tr>
    <td>Edges</td>
    <td>7444</td>
    <td>14866</td>
    <td>120217</td>
    <td>100422</td>
  </tr>
 
  <tr>
    <td>Clustering</td>
    <td>0.091</td>
    <td>0.488</td>
    <td>0.163</td>
    <td>0.642</td>
  </tr>
</table>

In order to obtain full datasets, it is required to download additional files via _download.sh_. Final revision of files structure includes general and local parts of datasets.

## General part (common for all datasets)
- SSORC_CS_2010_2021_authors_edge_list.csv -
- SSORC_CS_2010_2021_authors_edges_papers_indices.csv - 
- SSORC_CS_2010_2021_authors_features.csv -
- SSORC_CS_2010_2021_papers_features_vectorized_compressed_32.csv -

## Local part (unique for each dataset)
- <...>_authors_edges_papers_indices.csv - 
- SSORC_CS_10_21_1437_3164_unfiltered_authors.edgelist - 
- SSORC_CS_10_21_1437_3164_unfiltered_journals.csv - 
- SSORC_CS_10_21_1437_3164_unfiltered_papers.edgelist - 
- SSORC_CS_10_21_1437_3164_unfiltered_papers_targets - 

# Model running

You can run a following command to test REIGNN

```
python main.py
```

This command uses the sample dataset from _/datasets_ folder and receive as an input following args: 

You also can use REIGNN.py directly in your own experimental environment:


```python
from model.REIGNN import REIGNN
from utils import train

# description of
# input data

# define the model
model = REIGNN(data_citation, heads, train_data_a, val_data_a, test_data_a, authors_to_papers, 
                   cit_layers, latent_size_cit, auth_layers, latent_size_auth, link_size)
# train
epochs = 100
train(epochs)

# predict
prediction = [model.predict(t) for t in test_edges]
```

# Constructing your own dataset

We also publish our code regarding processing of initial datasets. It can be freely used for the construction of new heterogeneous graphs.

```python
from extractor import loader

# and so it goes

```

# Contact us

If you have some questions about the code, you are welcome to open an issue or send me an email, I will respond to that as soon as possible.

# Citation

```
To be added
```

