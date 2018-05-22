import numpy as np
import tensorflow as tf
import utils
import os
import scipy.io
import matplotlib.pyplot as plt
# from PIL import Image


class DatasetLoader():
    """Load fixed images, moving images and degrees.

    DataLoader defines basic operations to load data.
    the data should be organized in the following fashion:

    all according to MATLAB function GenerateDescriminatorData
    (Paths are supllied in PathFile.txt)

    Path:
    fixed_image.mat
    RotateImages
        -5Degrees.mat
        -4Degrees.mat
        ..
        ..
        5Degrees.mat


    """

    def __init__(self, config_path):
        """Construct a data loader using a configuration file.

        Args:
            config_path: path to .json configuration file.
            see __init__ definition to observe required configurations.

        Returns:
            a basic DataLoader instance.

        Raises:
            NameError: if configuration file path can't be found.
        
        Todo:
            add assertions for all parameters.
        
        """
        self.params = utils.import_json_config(config_path)
        self.crop_size = self.params["crop_size"]
        self.batch_size = self.params["batch_size"]
        self.num_of_epochs = self.params["num_of_epochs"]
        self.channels = self.params["channels"]
        self.image_size = self.params["image_size"]
        self.directory = os.getcwd()

    def _single_load(self, _dir):
        # TODO this function needs some cleaning
        Rot_list, image_list , degrees_list = self._image_pairs(_dir)
        Rot_files = np.ndarray([len(Rot_list), self.image_size[0], self.image_size[1], self.channels])

        image_files = np.ndarray([len(image_list), self.image_size[0], self.image_size[1], self.channels])
        idx = 0
        for Rot in Rot_files:
            if os.path.splitext(Rot)[1] == '.mat':
                Rot_files[idx, :, :, :] = np.expand_dims(scipy.io.loadmat(Rot), axis=2)
            elif self.channels == 3:
                pass
            else:
                pass
            idx += 1
        idx = 0
        for image in image_list:
            if os.path.splitext(image)[1] == '.mat':
                image_files[idx, :, :, :] = np.expand_dims(scipy.io.loadmat(image), axis=2)
            elif self.channels == 3:
                pass
            else:
                pass
            idx += 1
        image_tensor = tf.convert_to_tensor(image_files, dtype=np.float32)
        # TODO check if the float32 type reduce the resolution
        Rot_tensor = tf.convert_to_tensor(Rot_files, dtype=np.float32)
        Degrees_tensor = tf.convert_to_tensor(degrees_list, dtype=np.float32)
        # ranks_tensor = tf.reshape(ranks_tensor, [-1, 1])

        return image_tensor, Rot_tensor, Degrees_tensor

    def _spatial_augmentation(self, seg, image, rank):
        # concat seg and image together:
        batch = tf.concat(axis=2, values=[seg, image])
        if self.params["crop"]:
            batch = self._random_crop(batch)
        if self.params["flip"]:
            batch = tf.image.random_flip_up_down(batch)
            batch = tf.image.random_flip_left_right(batch)
        if self.params["rotate"]:
            batch = tf.image.rot90(batch, k=np.random.randint(0, 3))
            # separate seg and image:
        seg = batch[:, :, 0:self.params["channels"]]
        image = batch[:, :, self.params["channels"]:2 * self.params["channels"]]

        return seg, image, rank

    def _random_crop(self, batch):
        return tf.random_crop(batch, [self.crop_size[0], self.crop_size[1], self.params["channels"] * 2])

    @staticmethod
    def _ranks_reader(ranksdir, index, isdice=True):
        orig_dir = os.getcwd()
        os.chdir(ranksdir)
        ranks_list = []

        if isdice:
            ranks = utils.import_json_config('dice_ranks_%03.f.json' % index)
        else:
            ranks = utils.import_json_config('jaccard_ranks_%03.f.json' % index)

        for rank in ranks:
            ranks_list.append(ranks[rank])

        os.chdir(orig_dir)

        return ranks_list

    def _image_pairs(self, datadir):

        fix_image_list = []
        Rot_image_list = []
        label_list = []

        # load image names into a list
        imagedir = (self.directory + '/%s' % datadir)
        # TODO think if we don't want to get the absolute dir
        # in the function call (load function in origin)
        fix_image_list.append((imagedir + '/fix_image.mat'))
        current_dir1 = (imagedir + '/RotateImages')
        Rot_filenames = os.listdir(current_dir1)
        for Rot_image in Rot_filenames:
            Rot_image_list.append((current_dir1 + '/%s' % Rot_image))
        current_dir2 = (imagedir + '/labels')
        label_filenames = os.listdir(current_dir2)
        for label in label_filenames:
            Rot_image_list.append((current_dir2 + '/%s' % label))
        return Rot_image_list, fix_image_list , label_list

    def intelligent_load(self, dir_list):
        """Load images and segmentations intelligently, minimizing memory excess use.
        
        TODO: implement!
        1. single load will produce:
            (image0, seg00, seg01,..., seg0N, ranks00, ranks01,..., ranks0N)
            (image1, seg10, seg11,..., seg1N, ranks10, ranks11,..., ranks1N)
            (             ...               ,          ...                 )
            (imageM, segM0, segM1,..., segMN, ranksM0, ranksM1,..., ranksMN)
        2. a dataset will be created for each of the lists.
        3. every dataset will be shuffled with the same seed, maintaining ordering.
        4. two new dataset will be created from concatanating all of the segs and all of the ranks.
        5. the image dataset will go through "repeat", N times..
        6. the input_fn will recieve the three datasets. it will make one_shot_iterators from every dataset
        7. we will manually join the get_next() from the images and from the segs datasets into a dictionary,
            and input_fn will produce:
            dict("image": image_iter.get_next(), "seg": seg_iter.get_next), rank_iter.get_next(
        
        """
        raise NotImplementedError

    def tile_load(self, dir_list):
        """Load images and segmentation by repeating the images in memory for each segmentation.
        
        Tile load is not an optimal solution since the same data is replicated in the memory.
        We propose the following solution to be implemented later:

        At the moment we assume #images == #segs/ranks
        this means that the images are REPLICATED in the memory
        it is not ideal and better implementation will be to use intelligent_load()

        Args:
            dir_list: list of directories, each a 'sequence_dir' that is built
                according to the hierarchy defined in the DatasetLoader class.

        Todo:
            Currently we support only one dir at a time!
        
        """
        # we assume segs contains all segmentations for images, maintaining ordering
        # if there are N segs per image, we assume they are at places
        # (i, M+i, 2M+i... (N-1)*M+i) for image i
        
        images, Rots, degrees = self._single_load(dir_list[0])
        num_images = images.shape.as_list()[0]
        num_Rot = Rots.shape.as_list()[0]

        images = tf.tile(images, [np.int32(num_Rot / num_images), 1, 1, 1])
        degrees = tf.reshape(degrees, [-1])
        # will raise exception if number of images != number of segs
        return tf.data.Dataset.from_tensor_slices((images, Rots, degrees))

    def shuffle_batch_repeat(self, dataset, batch_size=None, repeat_num=None):
        """Shuffle, batch and repeat a dataset according to configuration.

        The configuration file defines the number of repeats (num_of_epochs)
        and the batch size.

        Args:
            dataset: A Dataset on which to perform the transformations.
            batch_size: Optional. configures the sizes of the batchs.
                Default taken from configuration.
            repeat_num: Optional. configures how many times to repeat the dataset.
                Default taken from configuration.

        Returns
            a Dataset.
        
        """
        if batch_size is None:
            batch_size = self.batch_size
        if repeat_num is None:
            repeat_num = self.num_of_epochs
        
        # TODO asaf - 128 is an arbitrary number for the buffer size in shuffle
        dataset = dataset.map(self._spatial_augmentation)
        return dataset.shuffle(128).batch(batch_size).repeat(repeat_num)

    @staticmethod
    def input_fn(dataset):
        """Organize the features and labels to be inserted into the model's model_fn.

        From Tensorflow's Programmer's Guide:
        A function that constructs the input data for evaluation.
        The function should construct and return one of the following:
        A 'tf.data.Dataset' object: Outputs of Dataset object must be a tuple
            (features, labels) with same constraints as below.
        A tuple (features, labels): Where features is a Tensor or a dictionary
            of string feature name to Tensor and labels is a Tensor
            or a dictionary of string label name to Tensor.
            Both features and labels are consumed by model_fn.
            They should satisfy the expectation of model_fn from inputs.

        Args:
            dataset: A Dataset that produces three elements: images, segs, ranks
        
        Returns:
            a get_next() operation of an Iterator on top of the new mapped dataset,
                after organizing the data in a (features, labels) tuple.
        
        """
        # change dataset structure to (features, labels)
        def parser(images, Rots, degrees):
            features = {"images": images, "Rots": Rots}
            labels = degrees
            return features, labels

        dataset = dataset.map(parser)
        return dataset.make_one_shot_iterator().get_next()


def main():
    pass
    # """Test the loaders, for debug purposes only."""
    # # current_dir = os.getcwd()

    # config_path = 'dataLoaderParams.json'
    # test = DatasetLoader(config_path)
    # tf.global_variables_initializer()
    # #seg, image, rank = test.tile_load(['data/TDG/fluo-c2dl-msc'])

    # with tf.Session() as sess:

    #     coord = tf.train.Coordinator()
    #     threads = tf.train.start_queue_runners(coord=coord)
    #     seg_batch, image_batch, rank_batch = sess.run([seg, image, rank])
    #     coord.request_stop()
    #     coord.join(threads)
    #     for i in range(0, np.shape(seg_batch)[0]):
    #         img = np.squeeze(image_batch[i, :, :, :], axis=2)
    #         seg = np.squeeze(seg_batch[i, :, :, :], axis=2)
    #         plt.subplot(1, 2, 1)
    #         plt.imshow(img)
    #         plt.subplot(1, 2, 2)
    #         plt.imshow(seg)
    #         plt.title("Seg Rank: %s" % rank_batch[i])
    #         plt.show()


if __name__ == '__main__':
    main()
