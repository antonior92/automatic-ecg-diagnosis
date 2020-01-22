# Model predictions on the test set
This folder contain the deep neural network predictions on the test set. All files are in
the format `.npy` and can be read using `numpy.load()`. Each one should contain a 

All the content within this folder can be generate using the following sequence of commands:

(without a GPU it should take about 25 minutes. With GPU acceleration it should take
less then one minute)

 ```bash
cd /path/to/automatic-ecg-diagnosis
PFOLDER="./dnn_predicts"
MFOLDER="./model"
DFOLDER="./data"

# To generate the predictions on the test set corresponding to the main model used allong the paper use:

python predict.py --tracings $DFOLDER/ecg_tracings.hdf5 --model $MFOLDER/model.hdf5 --output_file $PFOLDER/model.npy


# We also train several networks with the same architecture and configuration
# but with different initial seeds.  In order to generate the neural network 
# prediction on the test set for each of these models:

mkdir $FNAME/other_seeds
for n in 1 2 3 4 5 6 7 8 9 10
do
python predict.py --tracings $DFOLDER/ecg_tracings.hdf5 --model $MFOLDER/other_seeds/model_$n.hdf5 --output_file $PFOLDER/other_seeds/model_$n.npy
done


# Finally, to asses the effect of how we structure our problem, we have considered alternative s
# cenarios where we use 90\%-5\%-5\% splits, stratified randomly,
# by patient or in chronological order. The predictions of those models in the test set
# can be obtained using:

mkdir $PFOLDER/other_splits
for n in date_order individual_patients normal_order
do
python predict.py --tracings $DFOLDER/ecg_tracings.hdf5 --model $MFOLDER/other_splits/model_$n.hdf5 --output_file $PFOLDER/other_splits/model_$n.npy
done
```

Where the `DFOLDER` should give the path to the folder containing the test dataset and `MFOLDER` should point to the 
folder containing pre-trained models. The test dataset can be downloaded from [here](https://doi.org/10.5281/zenodo.3625006) and the
pretrained models can be downloaded from here [here](https://doi.org/10.5281/zenodo.3625017)