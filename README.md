# DGL2024 Brain Graph Super-Resolution Challenge

## Contributors

Team Name: LinkLogic
- Gita Salsabila
- Amani Aldahmash
- Kangle Yuan
- Mengyu Rao
- Jiqiu Hu

## Problem Description

Brain image quality can vary due to environmental conditions or imaging technologies. High-quality images provide detailed information but are challenging to obtain, while low-quality images lack detail but are more affordable. This issue has led to the development of methods like Graph Super-Resolution (GSR-Net) to enhance brain image quality. GSR-Net focuses on brain connectivity represented as graphs, where nodes represent brain regions and edges represent connections. By leveraging this connectivity, GSR-Net can capture complex relationships in graph data and improve low-resolution brain connectivity maps. In our research, we aimed to build an improved version of GSR to predict high-resolution connectivity graphs from low-resolution brain graphs in an inductive learning setting. Solving this problem is important because it enables us to gain more detailed insights into the complex network of the brain. This advancement could lead to better understanding of the neural connectivity patterns in general. 

## Att-GSR Methodology

Our Att-GSR model implements the GSR with two core enhancements. The first being a graph U-Autoencoder with GAT layers, which use attention mechanisms to understand node interactions and graph structures, optimising the multiscale representation without the need for padding. The second enhancement is a modified GSR layer that features an AdamW optimiser and dropout layers, improving generalisation and incorporating eigen-decomposition losses. The model’s training is fine-tuned with a tailored loss function along with a lightweight architecture that facilitates efficient learning over 200 epochs.

![ image info](pipeline.jpeg)

## Used External Libraries

We ran our model on Paperspace Gradient P5000 Machine (30 GiB RAM, 8-core CPU, and 16 GiB GPU NVIDIA Quadro) using several external libraries. To replicate our environment and run the code successfully, you need to install the following Python libraries:

- pandas
- numpy
- torch
- torch-geometric
- scikit-learn
- matplotlib
- networkx
- scipy

To do this,you can use the following commands:

```
!pip install torch==1.14.0
!pip install pandas>=1.2.4 
!pip install numpy>=1.18.5 
!pip install matplotlib>=3.2.2
!pip install networkx>=2.5
!pip install scipy>=1.6.3
!pip install scikit-learn>=0.22.2
!pip install torch-geometric==0.12.0
```


## Results
#### Evaluation Metrics on 3-fold Cross Validation
![ image info](bars.png)
#### Heatmaps Visualization on generated graph
![ image info](heatmaps.png)

## References

[1] W. Liu, D. Wei, Q. Chen, W. Yang, J. Meng, G. Wu, T. Bi, Q. Zhang, X.-N. Zuo, and J. Qiu. Longitudinal test-retest neuroimaging data from healthy young adults in southwest china. Scientific data, 4(1):1–9, 2017.<br>
[2] M. Isallari, I. Rekik. GSR-Net: Graph Super-Resolution Network for Predicting High-Resolution from Low-Resolution Functional Brain Connectomes, 2010.<br>
[3] P. Andr ́as Papp, K. Martinkus, L. Faber, R. Wattenhofer. DropGNN: Random Dropouts Increase the Expressiveness of Graph Neural Networks, 2021.<br>
[4] Veliˇckovi ́c, Petar, et al. ”Graph attention networks.”arXiv preprintarXiv:1710.10903(2017).
