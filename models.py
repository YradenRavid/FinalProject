import tensorflow as tf
import utils


class _BaseModel:
    """Define internal ops and required implementations for rib-cage based neural networks.

    The base model parses the configuration file passed to it on initialization,
    and defines the basic ops of the network such as the activation, the loss,
    the rib structure and the optimizer.

    """

    def __init__(self, config):
        """Parse basic configurations for the model.

        Args:
            config: Configuration file for the network architecture.

        Returns:
            A basic model instance.

        Raises:
            AssertionError: if data_format in the configuration is not NHWC.
            KeyError: if some configuration can not be parsed.

        """
        self.config = config
        self.in_training = None

        assert self.config['data_format'] == 'NHWC', \
            "only NHWC format is supported"
        if self.config['data_format'] == 'NHWC':
            self.config['concat_dim'] = 3
            self.config['batch_norm_axis'] = -1
        else:
            self.config['concat_dim'] = 1
            self.config['batch_norm_axis'] = 1

    def activation(self, inputs):
        if self.config['activation'] == 'relu':
            return tf.nn.relu(inputs, name='relu')
        elif self.config['activation'] == 'leaky_relu':
            return tf.nn.leaky_relu(
                inputs, self.config['leaky_relu_alpha'], name='leaky_relu')
        else:
            raise ValueError('unknown activation function')

    def basic_rib(self, scope_name, inputs, outputs):
        in_training = self.in_training
        conv_kernel_size = self.config['conv_kernel_size']
        conv_strides = self.config['conv_strides']
        batch_norm_axis = self.config['batch_norm_axis']
        pool_size = self.config['pool_size']
        pool_strides = self.config['pool_strides']

        with tf.variable_scope(scope_name):
            conv = tf.layers.conv2d(
                inputs, outputs, conv_kernel_size,
                conv_strides, name='conv')  # output size: (W-F+2P)/S + 1
            # TODO asaf - check why epsilon is different than default 1e-3
            batch_norm = tf.layers.batch_normalization(
                conv, batch_norm_axis, training=in_training,
                epsilon=1e-1, name='batch_norm')
            relu = self.activation(batch_norm)
            pool = tf.layers.max_pooling2d(
                relu, pool_size, pool_strides, name='max_pool')

        return pool

    def train_loss_op(self, ranks, predictions, huber_delta=0):
        if self.config['loss'] == 'huber':
            return tf.losses.huber_loss(ranks, predictions, delta=huber_delta)
        else:  # MSE
            return tf.losses.mean_squared_error(ranks, predictions)

    def mean_single_prediction(self, predictions):
        """Predict segmentation rank in case all samples are from the same image.
        
        In case we sample several crops from the same image, we can take the mean
        of the predictions as the prediction per one image

        Args:
            predictions: The predictions tensor for the ranks.

        Returns:
            A scalar tensor.
        
        """
        return tf.reduce_mean(predictions)

    def optimizer(self):
        return {
            'Adam': tf.train.AdamOptimizer(),
            'RmsProp': tf.train.RMSPropOptimizer(0.001)
            # Adam is the default optimizer
        }.get(self.config['optimizer'], tf.train.AdamOptimizer())

    def build(self):
        raise NotImplementedError

    @staticmethod
    def model_fn(features, labels, mode, params):
        """Build the network according the mode of operation and the given inputs.

        Implement according to Tensorflow Custom Estimators Guide:
        https://www.tensorflow.org/versions/master/api_docs/python/tf/estimator/Estimator

        """
        raise NotImplementedError


class RibCageRegressionNet(_BaseModel):
    """Define a regression model of ranking segmentations, on a continuous scale from 0 to 1.

    The RibCageRegressionNet is comprised of two main elemnts:
    
    rib_cage:
        A deep sequence of triplets of basic_rib_cage as defined in the BaseModel.
        The triplets form three deep queues: one for the segmentation frame,
        one for the image, and one that is a concatenation of the other two.
    
    sternum:
        A deep sequence of dense layers ending in one neuron prediction the rank
        of the segmentation.

    """

    def __init__(self, config_path):
        """Parse configurations for the RibCageNet architecture.
        
        Args:
            config_path: Configuration file path for the network architecture.

        Returns:
            An instance of the RibCageRegressionNet builder.

        Raises:
            AssertionError: if data_format in the configuration is not NHWC.
            KeyError: if some configuration can not be parsed.
            
        """
        config = utils.import_json_config(config_path)
        super(RibCageRegressionNet, self).__init__(config)

    def build(self, features, in_training, reuse):
        """Build the network according to the configuration.

        Args:
            features: The inputs to the network, given as a dictionary containing
                two keys: 'segs' and 'images', each contains the data in NHWC format.
                obviously, all dimensions must be equal.
            in_training: A boolean representing the state of the network.
                Critical to batch_norm and dropout layers.
            reuse: Reuse flag for reusing weights under the variable scopes
                of the rib_cage and the sternum.
        
        Returns:
            Segmentation rank predictions tensor for N samples.

        Raises:
            AssertionError: if sizes of images and segs are not equal.

        """
        self.in_training = in_training
        print('RibCageRegressionNet: Model was built with in training=' + str(in_training))

        left_ribs = []
        right_ribs = []
        center_ribs = []
        sternum = []

        segs = features['Rots']
        images = features['images']

        assert(segs.shape.as_list() == images.shape.as_list())

        with tf.variable_scope('rib_cage', reuse=reuse):
            self._left_input = segs
            self._right_input = images
            self._center_input = tf.concat(
                [self._left_input, self._right_input], self.config['concat_dim'])
            # rib 0
            outputs = self.config['conv_filters_list'][0]
            left_ribs.append(
                self.basic_rib('left_rib_0', self._left_input, outputs))
            right_ribs.append(
                self.basic_rib('right_rib_0', self._right_input, outputs))
            center_ribs.append(
                self.basic_rib('center_rib_0', self._center_input, outputs))

            # build list of ribs in a variable legth
            # and connect each rib's output to the next input

            concat_dim = self.config['concat_dim']
            for d in range(1, self.config['conv_depth']):
                outputs = self.config['conv_filters_list'][d]
                left_ribs.append(
                    self.basic_rib('left_rib_' + str(d), left_ribs[d - 1], outputs))
                right_ribs.append(
                    self.basic_rib('right_rib_' + str(d), right_ribs[d - 1], outputs))
                concat = tf.concat(
                    [left_ribs[d - 1], right_ribs[d - 1], center_ribs[d - 1]], concat_dim)
                center_ribs.append(
                    self.basic_rib('center_rib_' + str(d), concat, outputs))

        with tf.variable_scope('sternum', reuse=reuse):
            last_concat = tf.concat(
                [left_ribs[d], right_ribs[d], center_ribs[d]], concat_dim)
            dense_input = tf.layers.flatten(last_concat, name="flatten")
            units = self.config['fc_units_list'][0]
            sternum.append(
                tf.layers.dense(dense_input, units, name='fc_0'))

            # build list of dense layers in a variable legth
            # and connect each layers output to the next input

            for d in range(1, self.config['fc_depth']):
                units = self.config['fc_units_list'][d]
                sternum.append(
                    tf.layers.dense(sternum[d - 1], units, name='fc_' + str(d)))
            with tf.variable_scope('predictions'):
                self.predictions = tf.sigmoid(sternum[d], 'sigmoid')
            return self.predictions

    @staticmethod
    def model_fn(features, labels, mode, params):
        """Build the network according the mode of operation and the given inputs.

        The model_fn is used in accordance with the requirments in Tensorflow's
        Estimators package.
        
        From Tensorflow's Programmer's Guide:
        'The final step in creating a model function is to write branching code
        that implements prediction, evaluation, and training.
        The model function gets invoked whenever someone calls the Estimator's
        train, evaluate, or predict methods.'

        Args:
            features: This is the first item returned from the input_fn passed
                to train, evaluate, and predict. This should be a single Tensor or dict of same.
            labels: This is the second item returned from the input_fn passed
                to train, evaluate, and predict. This should be a single Tensor or dict of same
                (for multi-head models). If mode is ModeKeys.PREDICT, labels=None will be passed.
                If the model_fn's signature does not accept mode,
                the model_fn must still be able to handle labels=None.
            mode: Optional. Specifies if this training, evaluation or prediction. See ModeKeys.
            params: Optional dict of hyperparameters.
                Will receive what is passed to Estimator in params parameter.
                This allows to configure Estimators from hyper parameter tuning.
            config: Optional configuration object.
                Will receive what is passed to Estimator in config parameter,
                or the default config. Allows updating things in your model_fn based on configuration
                such as num_ps_replicas, or model_dir.
        
        Returns:
            EstimatorSpec
        
        """
        # parse params:
        config_path = params['config_path']

        model = RibCageRegressionNet(config_path)
        with tf.variable_scope('RibCageRegressionNet'):
            train_predictions = model.build(features, in_training=True, reuse=False)
            test_predictions = model.build(features, in_training=False, reuse=True)

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode, predictions=test_predictions)
        # Create loss and optimizer
        train_loss_op = model.train_loss_op(labels, train_predictions)
        optimizer = model.optimizer()

        # from: https://www.tensorflow.org/api_docs/python/tf/layers/batch_normalization
        # when training, the moving_mean and moving_variance need to be updated.
        # By default the update ops are placed in tf.GraphKeys.UPDATE_OPS,
        # so they need to be added as a dependency to the train_op:

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(train_loss_op, global_step=tf.train.get_global_step())

        # Create logging hooks
        log_hook = tf.train.LoggingTensorHook({'loss': train_loss_op}, every_n_iter=100)
        print(train_loss_op)

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=train_predictions,
            loss=train_loss_op,
            train_op=train_op,
            training_hooks=[log_hook]
        )


if __name__ == '__main__':
    from main import main
    main()
