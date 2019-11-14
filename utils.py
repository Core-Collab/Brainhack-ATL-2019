import numpy as np
import time
import os

import SimpleITK as sitk
from dltk.io.augmentation import *
from dltk.io.preprocessing import *
from tqdm import tqdm


def load_data(file_references, mode, params=None):
    data = {'features': [], 'labels': []}

    # We define a `read_fn` and iterate through the `file_references`, which
    # can contain information about the data to be read (e.g. a file path):
    for meta_data in tqdm(file_references):

        # Here, we parse the `subject_id` to construct a file path to read
        # an image from.
        subject_id = meta_data[0]
        data_path = '/media/data/Track_2/'
        t1_fn = os.path.join(data_path, '{}.nii.gz'.format(subject_id))

        if not os.path.exists(t1_fn):
            continue

        # Read the .nii image containing a brain volume with SimpleITK and get
        # the numpy array:
        sitk_t1 = sitk.ReadImage(t1_fn)
        t1 = sitk.GetArrayFromImage(sitk_t1)

        # Normalise the image to zero mean/unit std dev:
        t1 = whitening(t1)

        # Create a 4D Tensor with a dummy dimension for channels
        t1 = t1[..., np.newaxis]

        # Labels: Here, we parse the class *sex* from the file_references
        # \in [1,2] and shift them to \in [0,1] for training:
        gt_label = np.int32(meta_data[1])
        y = gt_label
        # sex = np.int32(meta_data[1]) - 1
        # y = sex

        # If training should be done on image patches for improved mixing,
        # memory limitations or class balancing, call a patch extractor
        if params['extract_examples']:
            images = extract_random_example_array(
                t1,
                example_size=params['example_size'],
                n_examples=params['n_examples'])

            # Loop the extracted image patches
            for e in range(params['n_examples']):
                data['features'].append(images[e].astype(np.float32))
                data['labels'].append(y.astype(np.int32))

        # If desired (i.e. for evaluation, etc.), return the full images
        else:
            data['features'].append(images)
            data['labels'].append(y.astype(np.int32))

    data['features'] = np.array(data['features'])
    data['labels'] = np.vstack(data['labels'])

    return data


# Timer helper class for benchmarking reading methods
class Timer(object):
    """Timer class
       Wrap a will with a timing function
    """

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.t = time.time()

    def __exit__(self, *args, **kwargs):
        print("{} took {} seconds".format(
            self.name, time.time() - self.t))
