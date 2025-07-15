# Adversarial Training based Adaptive Weighted Hybrid Network for Autism Spectrum Disorder Diagnosis
#### AWHN-AT is a novel graph-based framework for Autism Spectrum Disorder (ASD) using functional MRI (fMRI) data. The method integrates graph construction, adversarial contrastive learning, and a dynamic training strategy into a unified architecture. It constructs both original and augmented graphs, extracts latent representations through a Graph Convolutional Network (GCN), and jointly optimizes the model using cross-entropy loss and an adversarial contrastive loss. The contrastive loss helps suppress redundant information in graph data, enhancing the discriminative capacity of the learned features, while the cross-entropy loss guides the classification task. A dynamic weighting mechanism is employed to smoothly shift the training focus from representation learning to classifier learning, improving both training stability and performance. Extensive experiments across multiple ABIDE datasets demonstrate that AWHN-AT outperforms existing state-of-the-art methods in accuracy, generalization, and interpretability. Moreover, it effectively identifies critical brain regions and connectivity patterns associated with ASD, offering valuable insights into the neural mechanisms of ASD and potential applications in clinical diagnosis.
## Usage
Setup
The whole implementation is built upon [PyTorch](https://pytorch.org/) and [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/).

#### Recommended Environment

- Python 3.7.1  
- [PyTorch 1.9.1 + cu111](https://pytorch.org/get-started/previous-versions/)
- [DGL 0.8.1 (CUDA 11)](https://www.dgl.ai/pages/start.html)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)

#### Installation Example

```bash
# Install PyTorch (with CUDA 11.1)
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

# Install DGL (CUDA 11.1)
pip install dgl-cu111==0.8.1 -f https://data.dgl.ai/wheels/repo.html

# Install PyTorch Geometric and dependencies
pip install torch-geometric

# Install other dependencies
pip install scikit-learn numpy matplotlib
```

#### How to Run Classification
The entire training and evaluation pipeline is integrated in `main.py`.

#### Hyperparameters

Below are key hyperparameters used in this project. Some can be passed via command-line, others are defined within the code (e.g., during graph construction).

| Parameter                  | Description                                                                 | Default       |
|---------------------------|-----------------------------------------------------------------------------|---------------|
| `lr`                      | Learning rate for the optimizer                                            | `0.0005`      |
| `weight_decay`            | L2 regularization (weight decay)                                           | `0.001`       |
| `batch_size`              | Number of samples per training batch                                       | `32`          |
| `epochs`                  | Total number of training epochs                                            | `300`         |
| `n_neighbors_graph`       | Number of neighbors used in `create_knn_graph()` for original graph        | e.g., `70`    |
| `n_neighbors_augment`     | Number of neighbors used in `augment_graph()` for generating augmented graph | e.g., `10`     |


## Dataset

This project uses the [ABIDE-I dataset](http://fcon_1000.projects.nitrc.org/indi/abide/), which provides multi-site resting-state fMRI data for Autism Spectrum Disorder (ASD) research.

