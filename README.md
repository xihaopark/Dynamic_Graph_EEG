# Dynamic_Graph_Learner_for_EEG

---
## Data
I have prepared a toy dataset for you: https://drive.google.com/file/d/1b20iPP_LSQfwbCEWs8BdsQ_oKK4sxBEy/view?usp=drive_link


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
```bash
python ./data/resample_signals.py --raw_edf_dir <tusz-data-dir> --save_dir <resampled-dir>
```
where `<tusz-data-dir>` is the directory where the downloaded data are located, and `<resampled-dir>` is the directory where the resampled signals will be saved.

## Experiments
### Configurations
You can modify settings and training parameters by editing the 'args.py' file. 
This includes adjusting the task, model, number of epochs, learning rate, batch size, and other model training parameters. 
Alternatively, you can specify them during execution using flags like '--num_epochs'.

### RUN
To train and test, you can run: 
```bash
python main.py --dataset TUSZ --input_dir <resampled-dir> --raw_data_dir <tusz-data-dir> --save_dir <save-dir> --task detection --model_name evobrain --num_epochs 100 
```
where `<save-dir>` is the directory where the results are located.

### Baselines
You can also test baselines by specifying '--model_name' with 'BIOT', 'evolvegcn', 'dcrnn', 'lstm', or 'cnnlstm'."

---

---

## Tasks

**Task 1**  
In the `model` folder, you will find two files: **`DCRNNori`** and **`DCRNN`**.  
- **`DCRNNori`** is the original version of the model.  
- **`DCRNN`** is the modified version that includes a graph structure learner and an information bottleneck.  
Compare these two implementations to understand how the graph learner and information bottleneck are integrated into the original framework.

**Task 2**  
Use the **ToyDataset** to run the following models and record the results:  
1. All baseline models.  
2. The original version **`DCRNNori`**.  
3. The new version **`DCRNN`** with the added components.  

Collect and compare the performance metrics for each model.

**Task 3**  
Check out the repository [VIB-GSL](https://github.com/RingBDStack/VIB-GSL) to further explore the design of learnable graph structures and information bottlenecks.  
- Their approach differs from ours in some details (e.g., feature masking).  
- Try adapting their method to our model and compare the outcomes with our current approach.  
- The original paper is also provided in their repo; you can refer to it for additional insights.

**Task 4**  
Coming soon.

