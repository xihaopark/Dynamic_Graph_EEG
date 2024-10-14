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

---
## Data

Temple University Seizure Corpus (TUSZ) v1.5.2 dataset is publicly available [here](https://isip.piconepress.com/projects/tuh_eeg/).
Once your request form is accepted, you can access the dataset.

---

## Setup

You can install the required dependencies using pip.

```bash
pip install -r requirements.txt
```

---

## Preprocessing
The preprocessing step resamples all EEG signals to 200Hz, and saves the resampled signals in 19 EEG channels as `h5` files.

On terminal, run the following:
```
python ./data/resample_signals.py --raw_edf_dir <tusz-data-dir> --save_dir <resampled-dir>
```
where `<tusz-data-dir>` is the directory where the downloaded TUSZ v1.5.2 data are located, and `<resampled-dir>` is the directory where the resampled signals will be saved.

## Experiments
### Configurations
You can modify settings and training parameters by editing the 'args.py' file. 
This includes adjusting the task, model, number of epochs, learning rate, batch size, and other model training parameters. 
Alternatively, you can specify them during execution using flags like '--num_epochs'.

### RUN
To train and test, you can run: 
```
python main.py --dataset TUSZ --input_dir <resampled-dir> --raw_data_dir <tusz-data-dir> --save_dir <save-dir> --task detection --model_name evobrain --num_epochs 100 
```
where `<save-dir>` is the directory where the results are located.

### Baselines
You can also test baselines by specifying '--model_name' with 'BIOT', 'evolvegcn', 'dcrnn', 'lstm', or 'cnnlstm'."
