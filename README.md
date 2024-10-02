# EvoBrain
Dynamic Multi-channel EEG Graph Modeling for Time-evolving Brain Network

## Abstract
We describe a novel dynamic graph neural network (GNN) approach for seizure detection and prediction from multi-channel Electroencephalography (EEG) data thet addresses several limitations of existing methods. While deep learning models have achieved notable success in automating seizure detection, static graph-based methods fail to capture the evolving nature of brain networks, especially during seizure events.

To overcome this, we propose EvoBrain, which uses a time-then-graph strategy that first models the temporal dynamics of EEG signals and graphs, and then employs GNNs to learn evolving spatial EEG representations. 
Our contributions include 
- (a) a theoretical analysis proving the expressivity advantage of time-then-graph over other approaches, 
- (b) a simple and efficient model that significantly improves AUROC and F1 scores compared with state-of-the-art methods, and 
- (c) the introduction of dynamic graph structures that better reflect transient changes in brain connectivity. 

We evaluate our method on the challenging early seizure prediction task. The results show improved performance, making EvoBrain a valuable tool for clinical applications. 
