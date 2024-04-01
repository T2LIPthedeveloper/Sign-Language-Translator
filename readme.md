# README 

## Setup


1. Download dataset from \href{https://www.kaggle.com/datasets/darthpenguin/merged-asl-alphabets/data}{Dataset Link}

2. Move the downloaded dataset into the project folder, ensuring the dataset folder contains "Test_Alphabet", "Train_Alphabet" and "alphabet.jpeg".

3. Run code using the below steps

### Run in Terminal
Create virtual env

```
python3.9 -m venv venv 
```
Activate Virtual Env
(mac)
```
source venv/bin/activate
```
(windows command prompt)
```
- venv\textbackslash Scripts\textbackslash activate 
```
Install required packages
```
pip install -r requirements.txt
```

### Run training.ipynb Jupyter Notebook
Run the cells in the notebook sequentially. Update in the second cell the dataset1 variable with "./{dataset folder name}/Train_Alphabet". The commented out sections in the first few cells are used if you would like to train the model on more datasets and for pre-processing. In the third last cell, update the test_path variable to be "./{dataset folder name}/Test_Alphabet"

Running the entire notebook will end with the test accuracies for both PointNet Models where we can see both accuracy values and any misclassified images will be saved into the misclassified folder. You may rename the destination folders defined in the predict_images() function for the misclassified files.


## Application
### Setup
1. Ensure the virtual environment is activated.

2. Set the model_path variable to be the path of the model you would like to use, this is in app.py

3. Start the application with 
```
python3 app.py
```
4. Place hand within video frame for a few seconds in desired letter hand sign. 

5. Predicted letter will appear on screen.


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


