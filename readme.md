# README 

## Setup

In terminal
```
python3.9 -m venv venv 
```
```
source venv/bin/activate
```
```
pip install -r requirements.txt
```

Download Dataset at the following link:
```
https://www.kaggle.com/datasets/darthpenguin/merged-asl-alphabets/data
```

Place the dataset folder into this project folder and rename dataset_1 in training.ipynb 



### Files

#### functions.py
```
This file defines functions to be used for dataset generation, dataset loading, training, validation
```
#### PointCNN.py
```
Defines the model using PointCNN architecture 
```
#### PointNet.py
```
Defines the model using PointNet architecture
```
#### training.ipynb
```
This file is to load use the functions from functions.py to create the dataset and other steps until testing
When accessing this file, select the venv as the kernel.
```
#### app.py 
```
This file uses a the stored models and runs an application.
set model_path to be the path of the model you would like to use.
```