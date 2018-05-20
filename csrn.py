import os
from os.path import abspath

from models import RibCageRegressionNet
from loaders import DatasetLoader

import tensorflow as tf
import argparse


def main(argv):
    """Run the csrn according to the arguments passed in the command line.

    Args:
        argv: A Namespace object. Possible returned from
            parser.parse_args() operation.
    
    """
    # fix osx warning message:
    # tensorflow/core/platform/cpu_feature_guard.cc:140]
    # Your CPU supports instructions that this TensorFlow binary wasnot compiled to use: AVX2 FMA
    if os.name == 'posix':
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    print(argv)

    # parse argv
    # set logging verbosity
    datadirs = [argv.datadirs]
    if argv.verbose:
        tf.logging.set_verbosity(tf.logging.INFO)
    if argv.model_dir is not None:
        model_dir = argv.model_dir
    else:
        model_dir = abspath('model/')
    if argv.config_path is not None:
        config_path = abspath(argv.config_path)
    else:
        config_path = abspath('config.json')
    # set seed
    if argv.seed is not None:
        tf.set_random_seed(argv.seed)
    if not argv.test:
        train = True
    
    # build estimator and loader
    model = tf.estimator.Estimator(
        RibCageRegressionNet.model_fn, model_dir=model_dir, params={"config_path": config_path})
    loader = DatasetLoader(config_path)
    if train:
        dataset = loader.tile_load(datadirs[0])  # datadirs - Path that is defines in PathFile (MATLAB)
        dataset = loader.shuffle_batch_repeat(dataset)

        for (idx, _dir) in enumerate(datadirs):
            # TODO: train + evaluation!
            try:
                model.train(lambda: loader.input_fn(dataset))
            except tf.errors.OutOfRangeError:
                if idx == 0:
                    continue
                dataset = loader.tile_load(_dir)
                dataset = loader.shuffle_batch_repeat(dataset)


if __name__ == '__main__':
    # Parse input arguments
    parser = argparse.ArgumentParser('csrn.py')

    # first positional argument: data directory
    parser.add_argument('datadirs', help='data directories list', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='print log outputs (default: false)')
    parser.add_argument('-m', '--model-dir',
                        help='model directory (default: temporary new folder)')
    parser.add_argument('-c', '--config-path',
                        help='config json file')
    parser.add_argument('--test', action='store_true',
                        help='invoke the csrnet in test mode (default: train)')
    parser.add_argument('-s', '--seed', type=int,
                        help='random seed initializer (default: random)')
    
    argv = parser.parse_args()
    main(argv)
