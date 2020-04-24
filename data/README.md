# Annotated 12 lead ECG dataset

Contain 827 ECG tracings from different patients, annotated by several cardiologists, residents and medical students.
It is used as test set on the paper:
"Automatic diagnosis of the 12-lead ECG using a deep neural network".
 https://www.nature.com/articles/s41467-020-15432-4.

It contain annotations about 6 different ECGs abnormalities:
- 1st degree AV block (1dAVb);
- right bundle branch block (RBBB);
- left bundle branch block (LBBB);
- sinus bradycardia (SB);
- atrial fibrillation (AF); and,
- sinus tachycardia (ST).

Companion python scripts are available in:
https://github.com/antonior92/automatic-ecg-diagnosis

--------

Citation
```
Ribeiro, A.H., Ribeiro, M.H., Paixão, G.M.M. et al. Automatic diagnosis of the 12-lead ECG using a deep neural network.
Nat Commun 11, 1760 (2020). https://doi.org/10.1038/s41467-020-15432-4
```

Bibtex:
```
@article{ribeiro_automatic_2020,
  title = {Automatic Diagnosis of the 12-Lead {{ECG}} Using a Deep Neural Network},
  author = {Ribeiro, Ant{\^o}nio H. and Ribeiro, Manoel Horta and Paix{\~a}o, Gabriela M. M. and Oliveira, Derick M. and Gomes, Paulo R. and Canazart, J{\'e}ssica A. and Ferreira, Milton P. S. and Andersson, Carl R. and Macfarlane, Peter W. and Meira Jr., Wagner and Sch{\"o}n, Thomas B. and Ribeiro, Antonio Luiz P.},
  year = {2020},
  volume = {11},
  pages = {1760},
  doi = {https://doi.org/10.1038/s41467-020-15432-4},
  journal = {Nature Communications},
  number = {1}
}
```
-----


## Folder content:

- `ecg_tracings.hdf5`:  this file is not available on github repository because of the size. But it can be downloaded
[here](https://doi.org/10.5281/zenodo.3625006). The HDF5 file containing a single dataset named `tracings`. This dataset is a 
`(827, 4096, 12)` tensor. The first dimension correspond to the 827 different exams from different 
patients; the second dimension correspond to the 4096 signal samples; the third dimension to the 12
different leads of the ECG exams in the following order:
 `{DI, DII, DIII, AVL, AVF, AVR, V1, V2, V3, V4, V5, V6}`.

The signals are sampled at 400 Hz. Some signals originally have a duration of 
10 seconds (10 * 400 = 4000 samples) and others of 7 seconds (7 * 400 = 2800 samples).
In order to make them all have the same size (4096 samples) we fill them with zeros
on both sizes. For instance, for a 7 seconds ECG signal with 2800 samples we include 648
samples at the beginning and 648 samples at the end, yielding 4096 samples that are them saved
in the hdf5 dataset. All signal are represented as floating point numbers at the scale 1e-4V: so it should
be multiplied by 1000 in order to obtain the signals in V.

In python, one can read this file using the following sequence:
```python
import h5py
with h5py.File(args.tracings, "r") as f:
    x = np.array(f['tracings'])
```

- The file `attributes.csv` contain basic patient attributes: sex (M or F) and age. It
contain 827 lines (plus the header). The i-th tracing in `ecg_tracings.hdf5` correspond to the i-th line.
- `annotations/`: folder containing annotations csv format. Each csv file contain 827 lines (plus the header).
The i-th line  correspond to the i-th tracing in `ecg_tracings.hdf5` correspond to the in all csv files.
The  csv files  all have 6 columns `1dAVb, RBBB, LBBB, SB, AF, ST`
corresponding to weather the annotator have detect the abnormality in the ECG (`=1`) or not (`=0`).
  1. `cardiologist[1,2].csv` contain annotations from two different cardiologist.
  2. `gold_standard.csv` gold standard annotation for this test dataset. When the cardiologist 1 and cardiologist 2
  agree, the common diagnosis was considered as gold standard. In cases where there was any disagreement, a 
  third senior specialist, aware of the annotations from the other two, decided the diagnosis. 
  3. `dnn.csv` prediction from the deep neural network described in 
  "Automatic Diagnosis of the Short-Duration12-Lead ECG using a Deep Neural Network". THe threshold is set in such way 
  it maximizes the F1 score.
  4. `cardiology_residents.csv` annotations from two 4th year cardiology residents (each annotated half of the dataset).
  5. `emergency_residents.csv` annotations from two 3rd year emergency residents (each annotated half of the dataset).
  6. `medical_students.csv` annotations from two 5th year medical students (each annotated half of the dataset).
