import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from PARAMS import *

from utils import *

# Set up parameters
batch_size = 5
iterations = 100

# Define the desired shapes and types of the training examples to pass to `read_fn`:
reader_params = {'n_examples': 1,
                 'example_size': RESIZE_SIZE,
                 'extract_examples': True}

reader_example_shapes = {'features': {'x': reader_params['example_size'] + [1, ]},
                         'labels': {'y': []}}

reader_example_dtypes = {'features': {'x': tf.float32},
                         'labels': {'y': tf.int32}}

all_filenames = pd.read_csv(
    '/media/data/Track_2/New_Labels_For_Track_2.csv',
    dtype=object,
    keep_default_na=False,
    na_values=[]).as_matrix()

# Load all data into memory
data = load_data(all_filenames,
                 tf.estimator.ModeKeys.TRAIN, reader_params)

x = tf.placeholder(reader_example_dtypes['features']['x'],
                   [None, 182, 218, 182, 1])
y = tf.placeholder(reader_example_dtypes['labels']['y'],
                   [None, 1])

dataset = tf.data.Dataset.from_tensor_slices((x, y))
dataset = dataset.repeat(None)
dataset = dataset.batch(batch_size)
dataset = dataset.prefetch(1)

features = data['features']
labels = data['labels']
# Check that features and labels dimensions match
assert features.shape[0] == labels.shape[0]

iterator = dataset.make_initializable_iterator()
nx = iterator.get_next()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.train.MonitoredTrainingSession(config=config) as sess_dict:
    # Initialize iterator
    sess_dict.run(iterator.initializer,
                  feed_dict={x: features, y: labels})

    with Timer('Feed dictionary'):
        # Timed feed dictionary example
        for i in range(iterations):
            # Get next features-labels pair
            dict_batch_feat, dict_batch_lbl = sess_dict.run(nx)

# Visualise the `dict_batch_feat` using matplotlib.
input_tensor_shape = dict_batch_feat.shape
center_slices = [s // 2 for s in input_tensor_shape]

# Visualise the `gen_batch_feat` using matplotlib.
f, axarr = plt.subplots(1, input_tensor_shape[0], figsize=(15, 5))
f.suptitle('Visualisation of the `dict_batch_feat` input tensor with shape={}'.format(input_tensor_shape))

for batch_id in range(input_tensor_shape[0]):
    # Extract a center slice image
    img_slice_ = np.squeeze(dict_batch_feat[batch_id, center_slices[1], :, :, :])
    img_slice_ = np.flip(img_slice_, axis=0)

    # Plot
    axarr[batch_id].imshow(img_slice_, cmap='gray')
    axarr[batch_id].axis('off')
    axarr[batch_id].set_title('batch_id={}'.format(batch_id))

f.subplots_adjust(wspace=0.05, hspace=0, top=0.8)
plt.show()
