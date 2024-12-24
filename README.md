# FAST
Code for paper names: Decoding Covert Speech from EEG Using a Functional Areas Spatio-Temporal Transformer (FAST)

### Reproduce 

To reproduce the result on BCI Competition 2020 Track #3: Imagined Speech Classification (BCIC2020Track3)

## Dataset Preparation
1. Download the dataset from [here](https://osf.io/pq7vb/).
2. Place the dataset in the `BCIC2020Track3/` directory.

### Path Structure
```
BCIC2020Track3/
├── Training set/
│   ├── Data_Sample01.mat
│   ├── Data_Sample02.mat
├── Validation set/
│   ├── Data_Sample01.mat
│   ├── Data_Sample02.mat
```

## Data Preprocessing
Run the following command to preprocess the data:
```bash
python dataset_BCIC2020Track3.py
```
The processed data will be saved in:
```
Processed/BCIC2020Track3.h5
```

## Training the Model
- To train the model, run:
```bash
python train_FAST_BCIC2020Track3.py
```
- If you have more than one GPU, you can use the following command instead:
```bash
bash run_dual_GPU.sh
```

## Evaluating the Results
- After training, results will be saved in the `Results/FAST` directory.
- To calculate the accuracy, run:
```bash
python calc_accuracy.py
```
