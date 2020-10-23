# DiffGCNs.py
DiffGCN: Graph Convolutional Networks via Differential Operators and Algebraic Multigrid Pooling

Code dependencies:
pytorch
torch-scatter
torch-sparse
torch-cluster
pytorch-geometric 
numpy


partsegmentation_train_test.py - experiment of part segmentation on shapenet
 		 Aero  Bag   Cap   Car  Chair Ear   Guit Knife  Lamp  Lapt  Motor  Mug  Pistol Rocket Skat  Table       Mean
 Ours   	 85.1  83.1  87.2  80.9 90.9  79.8  92.1  87.8  85.2  96.3   76.6  95.8  84.2  61.1   77.5  83.6  	86.4   
 Ours (pooling)  85.1  83.7  88.0  80.3  91.1  80.0  92.0  87.5  85.3  95.8  76.0  95.9  83.8  65.6   77.3  83.7  	86.4 

metric is mean intersection over union

To run the experiment, simply run python partsegmentation_train_test.py 
