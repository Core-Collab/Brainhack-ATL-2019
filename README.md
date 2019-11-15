# Automated Quality Control Tool to Identify Poorly Preprocessed MRI Scans

We developed a machine learning model that can identify errors in brain normalization and differentiate between properly and poorly preprocessed MRI scans.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

```
pip install tensorflow-gpu==1.14
pip install sklearn
pip install pandas
pip install numpy
pip install dltk
pip install tqdm
pip install jupyter
pip install torch
pip install torchvision
pip install scipy
pip install scikit-learn
pip install matplotlib
pip install pillow
```

### A small fix

You will need to fix your DLTK. We have included the edited file, `abstract_reader.py`. 
Simply run something like `cp abstract_reader.py /home/user/anaconda2/envs/env_name/lib/python3.6/dltk/io/`


### Paths to change

Our main runfile is `small_mains.py`. This has a call to the feature embedder, which needs fixed files. On line 11 you'll see:
```call_zac(txt_path='/media/data/Track_2/test_txt.txt', model_path='/media/data/models/brain/sl_resnet3d/new_backup3/')```
`txt_path` should be a path to the txt with absolute paths (separated by a newline, no commas) and `model_path` should be `new_backup3` in the repo where this is downloaded.

### Running

In a terminal, just run `python small_mains.py` and you'll see a ton of text dumped out. When the process completes, a new text file `small_scores.txt` will contain all predictions, continuous values between 0-1.

## Authors

* **Letian Chen**
* **Andrew Silva**
* **Manisa Natarajan**
* **Danny Comer**
* **Aniruddha Murali**

## Acknowledgments

* BrainHack ATL for hosting the neuroscience hackathon and providing us data