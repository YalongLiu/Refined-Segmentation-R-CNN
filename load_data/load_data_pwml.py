# coding:utf-8
import os
from skimage.io import imread
import numpy as np

from mrcnn import utils
from mrcnn import self_utils


class TargetDataset(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """

    def __init__(self, config):
        super(TargetDataset, self).__init__()
        self.split_dataset(config)

    def split_dataset(self, config):
        # split folders to train/val/test
        target_ids = next(os.walk(config.GT_DIR))[1]  # load in folders
        len_samples = len(target_ids)
        train_num = round(config.TRAIN_VAL_TEST_RATIO[0] * len_samples)
        val_num = round(config.TRAIN_VAL_TEST_RATIO[1] * len_samples)
        self.train_ids = target_ids[:train_num]
        self.val_ids = target_ids[train_num: train_num + val_num]
        self.test_ids = target_ids[train_num + val_num:]
        # print("train_ids:\t" + str(train_ids) + "\nval_ids:\t" + str(val_ids) + "\ntest_ids:\t" + str(test_ids))

    def compute_image_mean(self, config):
        images = []
        target_ids = next(os.walk(config.GT_DIR))[1]  # load in folders
        for sample_id in target_ids:
            mask_ids = next(os.walk(config.GT_DIR + '/' + sample_id))[2]
            for mask_id in mask_ids:
                image = imread(config.IMAGE_DIR + '/' + sample_id + '/' + mask_id)
                images.append(np.mean(image))
        mean_value = np.mean(images)
        return mean_value

    def load_samples(self, flag, config):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        self.add_class("pwml_t1_data", 1, "pwml")

        # Add images
        if flag == 'train':
            self.sample_ids = self.train_ids
        if flag == 'val':
            self.sample_ids = self.val_ids
        if flag == 'test':
            self.sample_ids = self.test_ids
        for sample_id in self.sample_ids:
            mask_ids = next(os.walk(config.GT_DIR + '/' + sample_id))[2]
            for mask_id in mask_ids:
                path = [config.INPUT_DIR, str(sample_id), str(mask_id)]
                self.add_image("pwml_t1_data", image_id=sample_id, path=path, image=None, mask=None)

    def sample_ids(self):
        return self.sample_ids

    def load_image(self, image_id, config):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but
        in this case it generates the image on the fly from the
        specs in image_info.
        """
        info = self.image_info[image_id]
        path_list = info['path']
        path = path_list[0] + '/images'
        for single in path_list[1:]:
            path = path + '/' + single
        image_ori = imread(path)
        if config.IMAGE_CHANNEL_COUNT == 3:
            image_ori = np.stack([image_ori, image_ori, image_ori], axis=-1)

        return image_ori

    def load_mask(self, image_id, config):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        path_list = info['path']
        path = path_list[0] + '/masks'
        for single in path_list[1:]:
            path = path + '/' + single
        mask_ori = imread(path)

        class_ids = []
        mask_flag = np.max(mask_ori)
        if mask_flag > 0:
            mask = self_utils.sementic2instance_mask(mask_ori)
            class_ids = np.ones([mask.shape[-1]], dtype=np.int32)  # Map class names to class IDs.
        else:
            assert 1 == 2, 'mask is all zero!'
            # mask = np.zeros(np.shape(mask_ori), dtype=np.uint8)
            # class_ids = np.array([0], dtype=np.int32)
        mask = mask.astype(np.bool)
        return mask, class_ids
