# Automated Quality Control Tool to Identify Poorly Preprocessed MRI Scans

We developed a machine learning model that can identify errors in brain normalization and differentiate between properly and poorly preprocessed MRI scans.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

```
pip install tensorflow-gpu==1.14 sklearn pandas numpy dltk tqdm jupyter torch torchvision scipy scikit-learn matplotlib pillow
```

### A small fix

You will need to fix your DLTK. We have included the edited file, `abstract_reader.py`. 
Simply run something like `cp abstract_reader.py /home/user/anaconda2/envs/env_name/lib/python3.6/dltk/io/`


### Paths to change

Our main runfile is `small_mains.py`. This has a call to the feature embedder, which needs fixed files. On line 11 you'll see:
```call_zac(txt_path='/media/data/Track_2/test_txt.txt', model_path='/media/data/models/brain/sl_resnet3d/new_backup3/')```
`txt_path` should be a path to the txt with absolute paths (separated by a newline, no commas) and `model_path` should be `new_backup3` in the repo where this is downloaded.

### Running

In a terminal, just run `python small_mains.py` and you'll see a ton of text dumped out. When the process completes, a new text file `small_scores.txt` will contain all predictions, continuous values between 0-1. Of course, this requires paths to be set appropriately and you need to have the necessary data

### What does each file do?

#### small_mains.py
The main runfile for our project, this file calls the 3D CNN to give us embeddings and scores, then uses those embeddings to get more scores, and finally averages over all of the scores for a final score that is then used in our evaluation. Being scored, you might say.

#### classifier.py
The sklearn training/saving/testing script, used for the classical ML estimators.

#### deploy.py
This script actually loads in the trained model and the given fMRI data, creating embeddings and predicting labels for a batch of inputs. It is called from the main script to generate embeddings for classical models and to give scores for final output.

#### train.py
The main training file for the 3D CNN, this creates the model in TensorFlow, loads in data from the reader, and begins to learn good vs. bad fMRI.

#### reader.py
Contains a Python read function for interfacing with `.nii.gz` filetypes, returning image data, id, and label.

#### PARAMS.py
Just houses a list

#### dataload_test.py
An example dataloader for TensorFlow.

#### utils.py
A utility file used in the dataload_test script.

#### clean_data.ipynb
Simple iPython notebook to create symlinks for all labeled examples in the dataset. If the label is "good", symlink the example to a "good" directory, and if it's bad then symlink it to a "bad" directory. Not necessary for the runfiles here, but a simple and potentially useful scipt.

#### pytorch_dataloader.ipynb
Again just a useful script that you won't need for this to run. If you want to get up and running with PyTorch later, this example data loader might be useful.

#### abstract_reader.py
Necessary fix to DLTK, see above.

## Authors

* **Letian Chen**
* **Andrew Silva**
* **Manisa Natarajan**
* **Danny Comer**
* **Aniruddha Murali**

## Acknowledgments

Thanks to the BrainHack ATL team for organizing the event, providing us with data/labels/food, and to the myriad of hosts who answered all of our questions and helped us understand the problem more clearly!
