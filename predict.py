# %% Import packages
import numpy as np
import warnings
import argparse
warnings.filterwarnings("ignore")
from keras.models import load_model
from keras.optimizers import Adam
import h5py

parser = argparse.ArgumentParser(description='Get performance on test set from hdf5')
parser.add_argument('--tracings', default="./ecg_tracings.hdf5",  # or date_order.hdf5
                    help='HDF5 containing ecg tracings.')
parser.add_argument('--model', default="./model.hdf5",  # or model_date_order.hdf5
                    help='file containing training model.')
parser.add_argument('--output_file', default="./dnn_output.npy",  # or predictions_date_order.csv
                    help='output csv file.')
parser.add_argument('-bs', type=int, default=32,
                    help='Batch size.')

args, unk = parser.parse_known_args()
if unk:
    warnings.warn("Unknown arguments:" + str(unk) + ".")


# %% Import
# Import data
with h5py.File(args.tracings, "r") as f:
    x = np.array(f['tracings'])
# Import model
model = load_model(args.model, compile=False)
model.compile(loss='binary_crossentropy', optimizer=Adam())
y_score = model.predict(x, batch_size=args.bs, verbose=1)

# Generate dataframe
np.save(args.output_file, y_score)

print("Model saved")