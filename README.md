# Film recommendation system
First steps in recommendation systems area
![](reports/figures/inference.gif)

## Description
This repository implemets the recomendation system based on the `MovieLens-100k` dataset. The repository contains the custom Graph Neural Networks trained on the dataset, benchmark model implementation.

Also, the inference for the recommendation is implemented. The user should enter his personal data, and then the model would generated the films he might like, based on the information from the dataset.

> Note: the custom GNN is still under development, so currenly only the benchmark model is implemented. 


## Structure
```bash
├── benchmark                 # Implementation of benchmarking
│   └── benchmark_models
│      └── GLocal_K.py        # Benchmark model training & evaluation
├── data                      # Directory for the data
├── models                    # Model weights
│   └── glocal_k
├── notebooks                 # Different exploration notebooks
│   ├── 01_data_visualization.ipynb    
│   ├── 02_handcrafted_gnn.ipynb
│   └── 03_models_comparision.ipynb
├── reports                   # Reports & Figures
│   └── figures
└──── src                     # Source code
    ├── data               
    │   ├── load_dataset.py   # Scripts for loading the datasets
    └── inference.py          # Inference script
```

## Models
### Benchmark model -- GLocal_K
The benchmark model is "GLocal_K" ([paper](https://arxiv.org/pdf/2108.12184.pdf)). The model is, in fact, autoencoder. However, the trick is in training the model: Firstly, an auto encoder is pretrained with the local kernelised weight matrix, which transforms the data from one space into the feature space by using a 2d-RBF kernel. Then, the pre-trained auto encoder is fine-tuned with the rating matrix, produced by a convolution-based global kernel, which captures the 
characteristics of each item. This results in quite decent performance (Fig. 4).

<center>
    <figure>
    <img src="reports/figures/glocal_k.png" alt="my alt text"/>
    <figcaption>Figure 4. The schematic overview of Glocal_K architecture (source: [paper](https://arxiv.org/pdf/2108.12184.pdf)). </figcaption>
    </figure>
</center>

The weights could be found [here](https://drive.google.com/drive/folders/1_uMUOMg1IBYFQdbtek-CVcc1ge-6R0GA?usp=drive_link).

## Installation
```bash
git push https://github.com/domrachev03/text_detoxification
```
and make sure to install [dependencies](https://github.com/domrachev03/gnn_recommendation_system/blob/master/environment.yml).

## Dependencies
The main dependencies are:
1. `Pytorch`
2. `Pytorch-Graphical`
3. Utilities: `unzip`, `argparse`

For the convenience, all the dependencies are provided in the `requirements.txt` and `environment.yml`.

## Utilization
1. Loading the dataset
   Load the dataset
   ```bash
   python3 src/data/load_dataset.py -u https://files.grouplens.org/datasets/movielens/ml-100k.zip
   ```
2. `Glocal-k` 
   Train:
      ```bash
      python3 benchmark/benchmark_models/GLocal_K.py -p data/raw/ml-100k -w models/glocal_k/
      ```
   Evaluation: 
      ```bash
      python3 benchmark/benchmark_models/GLocal_K.py -e -p data/raw/ml-100k -w models/glocal_k/best_model_mae.pt
      ```
3. Inference
   ```bash
   python3 src/inference.py -b --new     
   ```
## Acknowledgement
I would like to thanks the team of Practical Machine Learning and Deep Learning course for such a great opportunity to explore the world of Natural Language Processing. Special thanks to Vladimir Ivanov and Maxim Evgragov for tutorship and answering my (usually stupid) questions.