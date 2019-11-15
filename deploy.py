# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import argparse
import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.contrib import predictor
from tqdm import tqdm

from dltk.io.augmentation import extract_random_example_array
from dltk.io.abstract_reader import Reader

from reader import read_fn
from PARAMS import *
from train import model_fn

READER_PARAMS = {'n_examples': 1,
                     'example_size': RESIZE_SIZE,
                     'extract_examples': True}
# N_VALIDATION_SUBJECTS = 28


def call_zac(txt_path, model_path):
    # Read in the csv with the file names you would want to predict on
    filepath = txt_path
    with open(filepath) as fp:
        all_filenames = fp.readlines()
    file_names = [x.strip() for x in all_filenames]
    # file_names = np.loadtxt(txt_path, dtype=np.str)
    # file_names = pd.read_csv(
    #     '/media/data/Track_2/New_Labels_For_Track_2.csv',
    #     dtype=object,
    #     keep_default_na=False,
    #     na_values=[]).as_matrix()

    # We trained on the first 4 subjects, so we predict on the rest
    # file_names = file_names[-N_VALIDATION_SUBJECTS:]

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    nn = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=model_path,
        params={"learning_rate": 1e-3},
        config=tf.estimator.RunConfig(session_config=config))

    reader_params = {'n_examples': 1,
                     'example_size': RESIZE_SIZE,
                     'extract_examples': True}
    reader_example_shapes = {'features': {'x': reader_params['example_size'] + [1]},
                             'labels': {'y': []}}
    reader = Reader(read_fn,
                    {'features': {'x': tf.float32},
                     'labels': {'y': tf.int32}})
    input_fn, qinit_hook = reader.get_inputs(
        file_references=file_names,
        mode=tf.estimator.ModeKeys.PREDICT,
        example_shapes=reader_example_shapes,
        batch_size=1,
        shuffle_cache_size=1,
        params=reader_params)

    features_to_save = []
    labels_to_save = []
    for i in tqdm(nn.predict(input_fn, hooks=[qinit_hook])):
        features_to_save.append(i['features'])
        labels_to_save.append(i['y_'])

    return np.array(features_to_save), np.array(labels_to_save)
    # np.savetxt('embeds.csv', np.array(features_to_save), delimiter=',', newline='\n')
    # np.savetxt('labels.csv', np.array(labels_to_save), delimiter=',', newline='\n')


def predict(args):
    # Read in the csv with the file names you would want to predict on
    file_names = pd.read_csv(
        args.csv,
        dtype=object,
        keep_default_na=False,
        na_values=[]).as_matrix()

    # We trained on the first 4 subjects, so we predict on the rest
    # file_names = file_names[-N_VALIDATION_SUBJECTS:]

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    nn = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=args.model_path,
        params={"learning_rate": 1e-3},
        config=tf.estimator.RunConfig(session_config=config))

    reader_params = {'n_examples': 1,
                     'example_size': RESIZE_SIZE,
                     'extract_examples': True}
    reader_example_shapes = {'features': {'x': reader_params['example_size'] + [1]},
                             'labels': {'y': []}}
    reader = Reader(read_fn,
                    {'features': {'x': tf.float32},
                     'labels': {'y': tf.int32}})
    input_fn, qinit_hook = reader.get_inputs(
        file_references=file_names,
        mode=tf.estimator.ModeKeys.EVAL,
        example_shapes=reader_example_shapes,
        batch_size=1,
        shuffle_cache_size=1,
        params=reader_params)

    features_to_save = []
    labels_to_save = []
    for i in tqdm(nn.predict(input_fn, hooks=[qinit_hook])):
        features_to_save.append(i['features'])
        labels_to_save.append(i['y_'])

    np.savetxt('embeds.csv', np.array(features_to_save), delimiter=',', newline='\n')
    np.savetxt('labels.csv', np.array(labels_to_save), delimiter=',', newline='\n')

    # From the model_path, parse the latest saved model and restore a
    # predictor from it
    # export_dir = \
    #     [os.path.join(args.model_path, o) for o in sorted(os.listdir(args.model_path))
    #      if os.path.isdir(os.path.join(args.model_path, o)) and o.isdigit()][-1]
    # print('Loading from {}'.format(export_dir))
    # my_predictor = predictor.from_saved_model(export_dir)

    # Iterate through the files, predict on the full volumes and compute a Dice
    # coefficient
    # accuracy = []
    # for output in read_fn(file_references=file_names,
    #                       mode=tf.estimator.ModeKeys.EVAL,
    #                       params=READER_PARAMS):
    #     t0 = time.time()
    #
    #     # Parse the read function output and add a dummy batch dimension as
    #     # required
    #     img = output['features']['x']
    #     lbl = output['labels']['y']
    #     test_id = output['img_id']
    #
    #     # We know, that the training input shape of [64, 96, 96] will work with
    #     # our model strides, so we collect several crops of the test image and
    #     # average the predictions. Alternatively, we could pad or crop the input
    #     # to any shape that is compatible with the resolution scales of the
    #     # model:
    #
    #     # num_crop_predictions = 4
    #     # crop_batch = extract_random_example_array(
    #     #     image_list=img,
    #     #     example_size=RESIZE_SIZE,
    #     #     n_examples=num_crop_predictions)
    #
    #     y_ = my_predictor.session.run(
    #         fetches=my_predictor._fetch_tensors['y_prob'],
    #         feed_dict={my_predictor._feed_tensors['x']: crop_batch})
    #
    #     # Average the predictions on the cropped test inputs:
    #     y_ = np.mean(y_, axis=0)
    #     predicted_class = np.argmax(y_)
    #
    #     # Calculate the accuracy for this subject
    #     accuracy.append(predicted_class == lbl)
    #
    #     # Print outputs
    #     print('id={}; pred={}; true={}; run time={:0.2f} s; '
    #           ''.format(test_id, predicted_class, lbl[0], time.time() - t0))
    # print('accuracy={}'.format(np.mean(accuracy)))


if __name__ == '__main__':
    features_final, labels_final = call_zac("/media/data/Track_2/test_txt.txt", "/media/data/models/brain/sl_resnet3d/new_backup3/")
    print(features_final.shape)
    print(labels_final)
    import sys
    sys.exit()
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Preprocessing Quality Control')
    parser.add_argument('--verbose', default=False, action='store_true')

    parser.add_argument('--model_path', '-p',
                        default='/media/data/models/brain/sl_resnet3d/test/')
    parser.add_argument('--csv', default='/media/data/Track_2/New_Labels_For_Track_2.csv')

    args = parser.parse_args()

    # Set verbosity
    if args.verbose:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
        tf.logging.set_verbosity(tf.logging.INFO)
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        tf.logging.set_verbosity(tf.logging.ERROR)

    # Call training
    predict(args)
