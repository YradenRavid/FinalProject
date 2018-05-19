# CSRNet

# Few conventions for starting:
1. All (all!) parameters in this project should be configurable. WE DO NOT HARDCODE PARAMETERS!
2. the reason for the that is that we might want to iterate over variables to figure out the best model for our needs.
3. parameters will be loaded using the json format.
4. commit messages should be meaningful.
5. each new feature will be opened in a new, featureName branch. when it is integrated into the main project it will be done by either 'merge --squash' (small features) or rebase --interactive (large features).

21/04/18 TODO
THIS WEEK:
# data (amit):
1. accumolate a lot of training data, make sure algorithm is robust and can load all,
  not only one dir, including loading several dirs while training.
2. consider using tensorflow.data.Dataset (not to load all data at once? best practice?)

# training (asaf):
1. dump stats to FileWriter
2. create trainer class and training schemes
3. consider using: tensorflow.estimator.Estimator (to manage all training, save and load points checkpoints etc.)
4. create test net, with batch_normalization aware

NEXT WEEK:
hyper-parameter optimization (mostly batch-size, loss function, learning rate, crop size)

EXPERIMENTS FOR LATER:
1. different nets
    dropout,
    FC changed to 1 * 1 convs,
    max_pool changed to strides
2. different net depths for sternum / ribcage
