# FAST
Code for paper named: Decoding Covert Speech from EEG Using a Functional Areas Spatio-Temporal Transformer (FAST), which is currently under review

This codebase is for reproducing the result on the public available dataset called BCI Competition 2020 Track #3: Imagined Speech Classification (BCIC2020Track3)

## Dataset Preparation
1. Download the dataset from [https://osf.io/pq7vb/](https://osf.io/pq7vb/).
2. Place the dataset in the `BCIC2020Track3/` directory, the correct file structure should be the same as below:
```
BCIC2020Track3/
├── Training set/
│   ├── Data_Sample01.mat
│   ├── Data_Sample02.mat
....
├── Validation set/
│   ├── Data_Sample01.mat
│   ├── Data_Sample02.mat
...
```

## Data Preprocessing
Run the following command to preprocess the data:
```bash
python dataset_BCIC2020Track3.py
```
The processed data will be saved in `Processed/BCIC2020Track3.h5`

## Training the Model
- To train the model, run:
```bash
python3 FAST_BCIC2020Track3_train.py --gpu 0 --folds "0-15"
```
- If you have more than one GPU, you can use the following command instead:
```bash
bash run_dual_GPU.sh
```

## Evaluating the Results
After training, results will be saved in the `Results/FAST` directory, to calculate accuracy, run:
```bash
python3 FAST_BCIC2020Track3_eval.py
```
