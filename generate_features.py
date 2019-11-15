# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import argparse
import os
import pandas as pd
import tensorflow as tf
import numpy as np

from models import convolutional_autoencoder_3d
from dltk.io.abstract_reader import Reader

from reader_manisha import read_fn

X_features = []

def reconstruct_fn(features, labels, mode, params):
    net_output_ops = convolutional_autoencoder_3d(
        inputs=features['x'],
        num_convolutions=2,
        num_hidden_units=1024,
        filters=(16, 32, 64, 128),
        strides=((2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)),
        mode=mode)


    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=net_output_ops,
            export_outputs={'out': tf.estimator.export.PredictOutput(net_output_ops)})

    return tf.estimator.EstimatorSpec(mode=mode,
                                      predictions=net_output_ops,
                                      eval_metric_ops=None)


def predict(args):
    np.random.seed(42)
    tf.set_random_seed(42)

    print('Start Prediction')

    # Parse csv files for file names
    # all_filenames = pd.read_csv(
    #     args.data_csv,
    #     dtype=object,
    #     keep_default_na=False,
    #     na_values=[]).as_matrix()

    filepath = args.data_csv

    with open(filepath) as fp:
        all_filenames = fp.readlines()

    all_filenames = [x.strip() for x in all_filenames]

    train_filenames = all_filenames
    # val_filenames = all_filenames[100:120]

    # Set up a data reader to handle the file i/o.
    reader_params = {'n_examples': 1,
                     'example_size': [160, 192, 160],
                     'extract_examples': False}
    reader_example_shapes = {'features': {'x': reader_params['example_size'] + [1, ]}}
    reader = Reader(read_fn, {'features': {'x': tf.float32}})

    # Get input functions and queue initialisation hooks for training and
    # validation data
    pred_fn, pred_qinit_hook = reader.get_inputs(
        file_references=train_filenames,
        mode=tf.estimator.ModeKeys.PREDICT,
        example_shapes=reader_example_shapes,
        batch_size=1,
        shuffle_cache_size=1,
        params=reader_params)
    #
    # val_input_fn, val_qinit_hook = reader.get_inputs(
    #     file_references=val_filenames,
    #     mode=tf.estimator.ModeKeys.EVAL,
    #     example_shapes=reader_example_shapes,
    #     batch_size=BATCH_SIZE,
    #     shuffle_cache_size=SHUFFLE_CACHE_SIZE,
    #     params=reader_params)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # Instantiate the neural network estimator
    nn = tf.estimator.Estimator(
        model_fn=reconstruct_fn,
        model_dir=args.model_path,
        params={"learning_rate": 0.01},
        config=tf.estimator.RunConfig(session_config=config))

    # nn.predict(
    #             input_fn=pred_fn,
    #             hooks=[pred_qinit_hook],
    #             predict_keys=None,
    #             yield_single_examples=True
    # )

    for i in nn.predict(input_fn=pred_fn,
                hooks=[pred_qinit_hook],
                predict_keys=None,
                yield_single_examples=True):
        # print(i['hidden_units'].shape)

        X_features.append(i['x_'])

    return X_features






def call_manisha(data_path, model_path):
    # Set up argument parser
    parser = argparse.ArgumentParser(description='autoencoder training script')
    parser.add_argument('--run_validation', default=True)
    parser.add_argument('--restart', default=False, action='store_true')
    parser.add_argument('--verbose', default=False, action='store_true')
    parser.add_argument('--cuda_devices', '-c', default='0')

    parser.add_argument('--model_path', '-p', default=model_path)
    parser.add_argument('--data_csv', default=data_path)

    args = parser.parse_args()

    # Set verbosity
    if args.verbose:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
        tf.logging.set_verbosity(tf.logging.INFO)
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        tf.logging.set_verbosity(tf.logging.ERROR)

    # GPU allocation options
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Handle restarting and resuming training
    if args.restart:
        print('Restarting training from scratch.')
        os.system('rm -rf {}'.format(args.model_path))

    if not os.path.isdir(args.model_path):
        os.system('mkdir -p {}'.format(args.model_path))
    else:
        print('Resuming training on model_path {}'.format(args.model_path))

    # Call pred
    X_features = predict(args)

    X_features = np.array(X_features)
    X_features = X_features[::2]
    # print(X_features.shape)

    return X_features

    # Save to csv
    # np.savetxt('encoder_features.csv', X_features, delimiter=',', newline='\n')

if __name__ =='__main__':
    call_manisha()



