# REIGNN | Version 1 

![Pipeline_image](model/images/recommender_pipeline_rev4.png#gh-light-mode-only)
![Pipeline_image](model/images/recommender_pipeline_rev4dm.png#gh-dark-mode-only)

Welcome to the official repo of the REIGNNv1 model -- GNN-based recommender system for scientific collaborations assessment. Here we present the source code for ISWC'22 paper "Recommendations Become Even More Useful:
Multi-task Extension of Scientific Collaborations Forecasting".

Vladislav Tishin, Artyom Sosedka, Natalia Semenova, Anastasia Martynova, [Vadim Porvatov](https://www.researchgate.net/profile/Vadim-Porvatov)

arXiv PDF: _to be added_.

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
We use common subgraph extracted from Semantic Scholar Corpus as the basis for all of our datasets. All papers belong to the period from January 1st 2010 to December 31st 2021 and related to the area of Computer Science. 

- _SSORC_CS_2010_2021_authors_edge_list.csv_ - common graph edge list.
- _SSORC_CS_2010_2021_authors_edges_papers_indices.csv_ - common table describing relations between edges in a co-authorship graph (collaborations) and nodes in a citation graph (papers).  
- _SSORC_CS_2010_2021_authors_features.csv_ - table with one-hot encoded authors' research interests.
- _SSORC_CS_2010_2021_papers_features_vectorized_compressed_32.csv_ - table with vectorized via Universal Sentence Encoder abstracts of papers.

## Local part (unique for each dataset)
- <...>_authors.edgelist_ - edge list of a dataset citations graph.
- <...>_papers.edgelist_ - edge list of a dataset co-authorship graph.
- <...>_authors_edges_papers_indices.csv_ - table describing relations between edges in a co-authorship graph (collaborations) and nodes in a citation graph (papers). 
- <...>_papers_targets.csv_ - target values for each auxiliary task regarding edges in a co-authorship graph.


# Model running

You can use REIGNN.py directly in your own experimental environment:


```python
import torch
import torch.nn as nn

from model.dataloader import get_data
from model.utils import run
from model.REIGNN import REIGNN

root_dir = '../'
dataset_name, split_name, split_number = 'CS1021small', '5_0.1', 0
citation_graph, train_data, val_data, test_data, authors_to_papers, batch_list_x, batch_list_owner = get_data(root_dir, dataset_name, split_name, split_number)

# Global
epochs_per_launch, lr = 15000, 0.001
device = 'cuda:0'

# Local
c_conv_num, c_latent_size, a_conv_num, a_latent_size = 2, 128, 3, 384
operator, link_size, heads = "hadamard", 128, 1 

# Multitask weights
mt_weights = [0.05, 0.05, 0.05, 0.05]

# W&B parameters
wandb_output, project_name, entity, group  = False, 'REIGNN', 'test_entity', 'test_group'

# define the model
model = REIGNN(citation_graph.to(device), heads, device,\
                            train_data.to(device), val_data.to(device), test_data.to(device),
                            authors_to_papers,
                            cit_layers = c_conv_num, latent_size_cit = c_latent_size,
                            auth_layers = a_conv_num, latent_size_auth = a_latent_size,
                            link_size = link_size).to(device) 

optimizer, criterion = torch.optim.Adam(model.parameters(), lr=lr), nn.L1Loss()
run(wandb_output, project_name, group, entity, mt_weights, model, optimizer, criterion, operator, batch_list_x, batch_list_owner, epochs_per_launch)

```

# Constructing your own dataset

We also publish our code regarding processing of initial datasets. It can be freely used for the construction of new heterogeneous graphs -

```python
from dataset_extractor.loader import load
from dataset_extractor.utils import parse_global_dataset, preprocessing, extract_subgraph

# load and unpack Semantic Scholar Open Research Corpus
load()

# global dataset parameters
areas = ['Computer Science', 'Biology']
year_start, year_end = 2018, 2021
global_dataset_name = 'test_dataset'

# generate and save global dataset to ./processed_data
global_dataset = parse_global_dataset(areas, year_start, year_end, global_dataset_name)
processed_data = preprocessing(global_dataset)

# local dataset parameters 
local_dataset_name, nodes_number = 'test_local_dataset', 5000

# generate and save local dataset to ./datasets/%local_dataset_name%
extract_subgraph(global_dataset, processed_data, local_dataset_name, nodes_number)

```

# Contact us

If you have some questions about the code, you are welcome to open an issue or send me an email, I will respond to that as soon as possible.

# License

Established code released as open-source software under the MIT license.

# Citation

```
To be added
```

