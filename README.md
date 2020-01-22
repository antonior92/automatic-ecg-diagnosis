# Automatic ECG diagnosis using a deep neural network
Scripts and modules for training and testing deep neural networks for ECG automatic classification.
Companion code to the paper "Automatic Diagnosis of the Short-Duration12-Lead ECG using a Deep Neural Network".


- ``train.py``: Script for training the neural network. To train the neural network run: 
```bash
$ python train.py PATH_TO_HDF5 PATH_TO_CSV
```
Pre-trained models obtained using such script can be downloaded from [here](https://doi.org/10.5281/zenodo.3625017)


- ``predict.py``: Script for generating the neural network predictions on a given dataset.
```bash
$ python predict.py --tracings PATH_TO_HDF5_ECG_TRACINGS --model PATH_TO_MODEL  --ouput_file PATH_TO_OUTPUT_FILE 
```
The folder `./dnn_predicts` contain the output obtained by applying this script to the models available in
[here](https://doi.org/10.5281/zenodo.3625017) to make the predictions on tracings from 
[this test dataset](10.5281/zenodo.3625006).


- ``generate_figures_and_tables.py``: Generate figures and tables from the paper "Automatic Diagnosis o
the Short-Duration12-Lead ECG using a Deep Neural Network". Make sure to execute the script from the root folder,
so all relative paths are correct. So first run:
```
$ cd /path/to/automatic-ecg-diagnosis
```
Then the script
 ```bash
$ python generate_figures_and_tables
```
It should generate the tables and figure in the folder `outputs/`

- ``model.py``: Auxiliary module that defines the architecture of the deep neural network.
To print a summary of the model  layers run:
```bash
$ python model.py
```