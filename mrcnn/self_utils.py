# coding:utf-8
import csv
import os
import skimage
from skimage.measure import label
import numpy as np
import time
import matplotlib.pyplot as plt
import cv2
import sys
import logging
from scipy import ndimage
from skimage.io import imread, imsave
from tqdm import tqdm
import pydicom as dcm
import pandas as pd
from skimage import transform
from skimage import exposure
import random

from mrcnn import seg_eval_utils
from mrcnn import visualize
from mrcnn import utils

time_list = []


def mask_pad(mask_ori, pad_size):
    tmp_expand_mask = np.zeros([mask_ori.shape[0], pad_size])
    mask_expand = np.hstack([tmp_expand_mask, mask_ori, tmp_expand_mask])
    tmp_expand_mask = np.zeros([pad_size, mask_expand.shape[1]])
    mask_expand = np.vstack([tmp_expand_mask, mask_expand, tmp_expand_mask])
    mask_expand = np.uint8(mask_expand)
    return mask_expand


def mask_pad_off(mask_ori, pad_size):
    mask_pad_off = mask_ori[pad_size:-pad_size, pad_size:-pad_size]
    return mask_pad_off


def crop_to_skull(image):
    '''
    Crop black space in the image
    :param image:
    :return:
    '''
    if len(image.shape) == 3:
        image = image[:, :, 0]
    shape_ori = image.shape
    x1 = 0
    x2 = 0
    y1 = 0
    y2 = 0
    cut_threshold = 1024
    for i in range(shape_ori[1]):
        if (np.sum(image[:, i]) > cut_threshold) & (x1 == 0):
            x1 = i
        if (np.sum(image[:, shape_ori[1] - i - 1]) > cut_threshold) & (x2 == 0):
            x2 = shape_ori[1] - i - 1

    for i in range(shape_ori[0]):
        if (np.sum(image[i, :]) > cut_threshold) & (y1 == 0):
            y1 = i
        if (np.sum(image[shape_ori[0] - i - 1, :]) > cut_threshold) & (y2 == 0):
            y2 = shape_ori[0] - i - 1
    return x1, x2, y1, y2


def generate_patchs(image_ori=None, semantic_mask=[], patch_size=64):
    image_ori_shape = list(np.shape(image_ori))
    image_patchs = []
    mask_patchs = []
    if not semantic_mask == []:
        instance_mask = sementic2instance_mask(semantic_mask)
        bboxes = utils.extract_bboxes(instance_mask)
        for bbox in bboxes:
            y1, x1, y2, x2 = bbox
            x = int(round(x1 + x2) / 2)
            y = int(round(y1 + y2) / 2)
            shift_pos_x = random.randint(-int(round((x2 - x1) / 2)), int(round((x2 - x1) / 2)))
            shift_pos_y = random.randint(-int(round((y2 - y1) / 2)), int(round((y2 - y1) / 2)))
            new_x = x + shift_pos_x
            new_y = y + shift_pos_y
            if new_x >= image_ori_shape[1] - int(patch_size / 2):
                x2 = image_ori_shape[1]
                x1 = image_ori_shape[1] - patch_size
            elif new_x <= int(patch_size / 2):
                x2 = patch_size
                x1 = 0
            else:
                x1 = new_x - int(patch_size / 2)
                x2 = new_x + int(patch_size / 2)
            if new_y >= image_ori_shape[0] - int(patch_size / 2):
                y2 = image_ori_shape[0]
                y1 = image_ori_shape[0] - patch_size
            elif new_y <= int(patch_size / 2):
                y2 = patch_size
                y1 = 0
            else:
                y1 = new_y - int(patch_size / 2)
                y2 = new_y + int(patch_size / 2)
            image_patchs.append(image_ori[y1:y2, x1:x2])
            mask_patchs.append(semantic_mask[y1:y2, x1:x2])
        return image_patchs, mask_patchs
    else:
        bboxs = []
        n_y = int(image_ori_shape[0] / patch_size) + 1
        n_x = int(image_ori_shape[1] / patch_size) + 1
        delta_y = image_ori_shape[0] / n_y
        delta_x = image_ori_shape[1] / n_x
        pos_y = [delta_y * i for i in range(n_y)]
        pos_x = [delta_x * i for i in range(n_x)]
        for y in pos_y:
            for x in pos_x:
                if x + patch_size > image_ori_shape[0]:
                    x1 = image_ori_shape[1] - patch_size
                    x2 = image_ori_shape[1]
                else:
                    x1 = int(round(x))
                    x2 = int(round(x)) + patch_size
                if y + patch_size > image_ori_shape[0]:
                    y1 = image_ori_shape[1] - patch_size
                    y2 = image_ori_shape[1]
                else:
                    y1 = int(round(y))
                    y2 = int(round(y)) + patch_size
                bboxs.append([y1, x1, y2, x2])
                image_patchs.append(image_ori[y1:y2, x1:x2])
        return image_patchs, bboxs


def merge_patches(image, mask_patches, bboxs):
    mask = np.zeros(np.shape(image)[:2])
    for i in range(len(bboxs)):
        y1, x1, y2, x2 = bboxs[i]
        mask[y1:y2, x1:x2] = np.where(mask[y1:y2, x1:x2] == 0, mask_patches[i],
                                      (mask[y1:y2, x1:x2] + mask_patches[i]) / 2)
    return mask


def subplot_images(input_tuple, input_tuple_str, subplot_shape, figsize=[10, 10]):
    '''

    :param input_tuple: [image1,image2,GT_mask]
    :param input_tuple_str: ['image1',image2','GT_mask']
    :param subplot_shape: [1,3]
    :return: None
    '''
    input_tuple_id = 0
    len_input_tuple = len(input_tuple)
    plt.figure(figsize=(figsize[0], figsize[1]))
    for i in range(subplot_shape[0]):
        for ii in range(subplot_shape[1]):
            input_tuple_id = input_tuple_id + 1
            if input_tuple_id > len_input_tuple:
                break
            plt.subplot(subplot_shape[0], subplot_shape[1], input_tuple_id)
            plt.title(input_tuple_str[input_tuple_id - 1])
            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            plt.axis('off')
            plt.imshow(input_tuple[input_tuple_id - 1])
    plt.show()


def subplot_image_mask(image, mask):
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.subplot(1, 2, 2)
    color = visualize.random_colors(1, bright=True)
    plt.imshow(visualize.apply_mask(image, mask, color=color[0], alpha=0.5))
    plt.show()


def merge_image_pred_gt(image, pred_mask, gt_mask, alpha=0.5):
    '''
    the gt mask are blue, the overlap area are green, the other mask are red
    :param image:
    :param pred_mask:
    :param gt_mask:
    :return: image after merge
    '''
    color_red = [1.0, 0.0, 0.0]
    color_green = [0.0, 1.0, 0.0]
    color_blue = [0.0, 0.0, 1.0]
    overlap_mask = np.logical_and(pred_mask, gt_mask)
    pred2_mask = np.logical_and(pred_mask, np.logical_not(gt_mask))
    gt2_mask = np.logical_and(gt_mask, np.logical_not(pred_mask))
    image2 = visualize.apply_mask(image, overlap_mask, color=color_green, alpha=alpha)
    image2 = visualize.apply_mask(image2, pred2_mask, color=color_red, alpha=alpha)
    image2 = visualize.apply_mask(image2, gt2_mask, color=color_blue, alpha=alpha)
    return image2


def otsu_mask(image_input):
    '''
    Using OTSU method to segment the input image
    :param image_input: original image
    :return:OTSU mask
    '''
    kernel1 = np.ones((5, 5), dtype=np.uint8)
    kernel2 = np.ones((5, 5), dtype=np.uint8)
    kernel3 = np.ones((20, 20), dtype=np.uint8)
    ret1, th1 = cv2.threshold(image_input, 0, 255, cv2.THRESH_OTSU)
    image_close = cv2.erode(cv2.dilate(th1, kernel1), kernel2)
    image_erode = cv2.erode(image_close, kernel3)
    image_output = np.logical_and(th1, image_erode)
    #     print('ret1:',ret1)
    #     figuresize = (10,10)
    #     plt.figure(figsize=figuresize)
    #     plt.subplot(2,2,1)
    #     plt.imshow(th1)
    #     plt.subplot(2,2,2)
    #     plt.imshow(image_close)
    #     plt.subplot(2,2,3)
    #     plt.imshow(image_erode)
    #     plt.subplot(2,2,4)
    #     plt.imshow(image_final)
    #     plt.show()
    return image_output


def sementic2instance_mask(semantic_mask):
    '''
    Convert semantic mask into instance mask
    :param semantic_mask:
    :return: instance mask
    '''
    mask_shape = list(np.shape(semantic_mask))
    mask_labeled, instance_num = label(semantic_mask, return_num=True)
    instance_mask = np.zeros([mask_shape[0], mask_shape[1], instance_num], dtype=np.uint8)
    for i in range(instance_num):
        instance_mask[:, :, i] = np.where(mask_labeled == i + 1, True, False)
    return instance_mask


###########################
#     Inference
###########################
def detect_targets(model, config, image_ori):
    '''
    detect the image_ori to generate image_enlarge, pred_enlarge,pred and enlarge GT to generate gt_enlarge
    '''
    # enlarge image
    image_ori_shape = list(np.shape(image_ori))
    if config.IMAGE_CHANNELS == 3:
        if len(image_ori_shape) == 2:
            image_ori = np.stack([image_ori, image_ori, image_ori], axis=-1)
    elif config.IMAGE_CHANNELS == 1:
        if len(image_ori_shape) == 3:
            image_ori = image_ori[:, :, 0]
    image_enlarge, window, scale, padding = utils.resize_image(image_ori,
                                                               min_dim=config.IMAGE_MIN_DIM,
                                                               max_dim=config.IMAGE_MAX_DIM,
                                                               padding=config.IMAGE_PADDING)

    # Run detection
    results = model.detect([image_enlarge], verbose=0)
    r = results[0]

    roi_length = len(r['rois'])
    #         print("rois:", roi_length)
    if roi_length == 0:
        image_shape = list(np.shape(image_enlarge))
        pred = np.zeros([image_shape[0], image_shape[1]])
    else:
        pred = r['masks'][:, :, 0]
        for i in range(roi_length - 1):
            pred = np.maximum(pred, r['masks'][:, :, i + 1])
    pred_enlarge = np.where(pred > 0, [255], [0])

    # resiez the pred mask to the original size
    if image_ori_shape[0] > image_ori_shape[1]:
        pred = pred_enlarge[:, window[1]:window[3]]
    else:
        pred = pred_enlarge[window[2]:window[4], :]
    pred = ndimage.zoom(pred, zoom=[1 / scale, 1 / scale], order=0)

    return r, image_enlarge, pred_enlarge, pred


def enlarge_mask(image_ori=[], pred=[], gt=[], config=None):
    if len(list(np.shape(image_ori))) == 2:
        image_ori = np.stack([image_ori, image_ori, image_ori], axis=-1)
    image_enlarge, window, scale, padding, crop = utils.resize_image(image_ori, min_dim=config.IMAGE_MIN_DIM,
                                                                     min_scale=config.IMAGE_MIN_SCALE,
                                                                     max_dim=config.IMAGE_MAX_DIM,
                                                                     mode=config.IMAGE_RESIZE_MODE)
    pred_enlarge = []
    image_enlarge_shape = np.shape(image_enlarge)
    if pred != []:
        pred_shape = np.shape(pred)
        # if len(list(np.shape(pred))) == 2:
        #     pred = np.stack([pred, pred, pred], axis=-1)
        # if np.shape(pred) != (config.IMAGE_MIN_DIM, config.IMAGE_MAX_DIM, 3):
        #     pred_enlarge = utils.resize_mask(pred, [config.IMAGE_MIN_DIM, config.IMAGE_MAX_DIM], padding)
        # else:
        #     pred_enlarge = pred
        # pred_enlarge = pred_enlarge[:, :, 0]
        if not np.max(pred) == 0:
            pred = pred / np.max(pred)
            if pred_shape[:2] != (config.IMAGE_MIN_DIM, config.IMAGE_MAX_DIM):
                if pred_shape[0] == pred_shape[1]:
                    pred = pred * 255
                    pred_enlarge = utils.resize_mask(pred, [config.IMAGE_MIN_DIM, config.IMAGE_MAX_DIM],
                                                     [(0, 0), (0, 0)])  # pred should be 0-255
                else:

                    pred_enlarge = utils.rescale_mask(pred, scale, padding)  # pred should be 0-1
            else:
                pred_enlarge = pred
        else:
            pred_enlarge = np.zeros(image_enlarge_shape[:2], np.uint8)
    else:
        # pred_enlarge = np.zeros(image_enlarge_shape[:2], np.uint8)
        pass
    gt_enlarge = []
    if gt != []:
        if len(list(np.shape(gt))) == 2:
            gt = np.stack([gt, gt, gt], axis=-1)
        if np.shape(gt) != (config.IMAGE_MIN_DIM, config.IMAGE_MAX_DIM, 3):
            gt_enlarge = utils.rescale_mask(gt, scale, padding)
        else:
            gt_enlarge = gt
        gt_enlarge = gt_enlarge[:, :, 0]
    return image_enlarge, pred_enlarge, gt_enlarge


def post_process(dataset_test, model, config, pred_folder_name):
    '''
    data{'image_path', 'gt_path', 'save_path'}
    function:
    Detect
    if gt_path!=[]-->+Show
    if save_path!=[]-->+Save pred
    '''
    # variables
    images = []
    len_data = len(dataset_test.image_ids)
    whole_path = []
    for i, image_id in tqdm(enumerate(dataset_test.image_ids), total=len_data):
        # Load image and run detection
        info = dataset_test.image_info[image_id]
        path_list = info['path']
        path = config.PRED_DIR + '/' + pred_folder_name
        if not os.path.exists(path):  # make dir if the save dir is not exist
            os.mkdir(path)
        for single in path_list[1:-1]:
            path = path + '/' + single
            if not os.path.exists(path):  # if the folder is not exist
                os.makedirs(path)
            path = path + '/' + path_list[-1]  # the whole path
            if not os.path.exists(path):  # if image is not exist
                # print('...', path_list[1:], '\t', i, '/', len_data)
                image = dataset_test.load_image(image_id, config)
                # image = imread('F:\datasets\PWML_T1_data\PWML_T1_cut_to_skull_8b\images/8/77.png')
                # image = np.stack([image, image, image], axis=-1)
                image, window, scale, padding, crop = utils.resize_image(image, min_dim=config.IMAGE_MIN_DIM,
                                                                         min_scale=config.IMAGE_MIN_SCALE,
                                                                         max_dim=config.IMAGE_MAX_DIM,
                                                                         mode=config.IMAGE_RESIZE_MODE)
                images.append(image)
                whole_path.append(path)
                if len(images) == config.BATCH_SIZE:
                    # Detect objects
                    results = model.detect(images, verbose=0)
                    # Detect, plot, save
                    for n, result in enumerate(results):
                        # Processing pred_enlarge_list
                        if 'rois' in result:
                            roi_length = len(result['rois'])
                            #         print("rois:", roi_length)
                            if roi_length == 0:
                                pred_enlarge = np.zeros([config.IMAGE_MAX_DIM, config.IMAGE_MAX_DIM],
                                                        dtype=bool)
                            else:
                                pred_enlarge = result['masks'][:, :, 0]
                                for i in range(roi_length - 1):
                                    pred_enlarge = np.maximum(pred_enlarge,
                                                              result['masks'][:, :, i + 1])  # instance to semantic
                        else:
                            pred_enlarge = result['masks']
                        pred = np.squeeze(pred_enlarge)
                        pred = np.array(pred, bool)
                        imsave(whole_path[n], pred)
                    images = []
                    whole_path = []


def generate_features(dataset, model, config):
    '''
    data{'image_path', 'gt_path', 'save_path'}
    function:
    Detect
    if gt_path!=[]-->+Show
    if save_path!=[]-->+Save pred
    '''
    # variables
    images = []
    masks = []
    len_data = len(dataset.image_ids)
    image_whole_path = []
    mask_whole_path = []
    for i, image_id in tqdm(enumerate(dataset.image_ids), total=len_data):
        # Load image and run detection
        info = dataset.image_info[image_id]
        path_list = info['path']
        image_path = config.GEN_DIR + '/images'
        mask_path = config.GEN_DIR + '/masks'
        if not os.path.exists(image_path):  # make dir if the save dir is not exist
            os.mkdir(image_path)
            os.mkdir(mask_path)
        for single in path_list[1:-1]:
            image_path = image_path + '/' + single
            mask_path = mask_path + '/' + single
            if not os.path.exists(image_path):  # if the folder is not exist
                os.mkdir(image_path)
                os.mkdir(mask_path)
            image_path = image_path + '/' + path_list[-1]  # the whole path
            mask_path = mask_path + '/' + path_list[-1]  # the whole path
            if not os.path.exists(image_path):  # if image is not exist
                # print('...', path_list[1:], '\t', i, '/', len_data)
                image = dataset.load_image(image_id, config)
                mask, class_ids = dataset.load_mask(image_id, config)
                images.append(image)
                masks.append(mask)
                image_whole_path.append(image_path)
                mask_whole_path.append(mask_path)
                if len(images) == config.BATCH_SIZE:
                    # Detect objects
                    results = model.detect(images, verbose=0)

                    image, window, scale, padding, crop = utils.resize_image(image, min_dim=config.IMAGE_MIN_DIM,
                                                                             min_scale=config.IMAGE_MIN_SCALE,
                                                                             max_dim=config.IMAGE_MAX_DIM,
                                                                             mode=config.IMAGE_RESIZE_MODE)

                    mask = utils.rescale_mask(mask, scale, padding)
                    whole_mask = mask[:, :, 0]
                    len_mask = list(np.shape(mask))[2]
                    for i in range(len_mask - 1):
                        whole_mask = np.maximum(whole_mask, mask[:, :, i + 1])
                    mask = whole_mask
                    # Detect, plot, save
                    for n, result in enumerate(results):
                        # Processing pred_enlarge_list
                        if 'rois' in result:
                            roi_length = len(result['rois'])
                            #         print("rois:", roi_length)
                            if roi_length == 0:
                                continue
                            else:
                                rois = result['rois']
                                feature_gens = result['masks']
                                for ii, roi in enumerate(rois):
                                    y1, x1, y2, x2 = roi
                                    gt_mask_out = mask[y1:y2, x1:x2]
                                    feature_gen = feature_gens[ii]
                                    imsave(mask_whole_path[n][:-4] + '_' + str(ii) + '.png', gt_mask_out)
                                    np.save(image_whole_path[n][:-4] + '_' + str(ii), feature_gen)

                    images = []
                    masks = []
                    image_whole_path = []
                    mask_whole_path = []


def evaluate(dataset_test, config):
    # Define classes
    from enum import Enum
    # Use enumerations to represent the various evaluation measures
    class OverlapMeasures(Enum):
        jaccard, dice, volume_similarity, false_negative, false_positive, sensitive, specificity = range(7)

    class SurfaceDistanceMeasures(Enum):
        hausdorff_distance, mean_surface_distance, median_surface_distance, std_surface_distance, max_surface_distance = range(
            5)

    # Evaluation
    info = dataset_test.image_info[0]
    path_list = info['path']
    sample_num = path_list[1]
    csv_file = next(os.walk(config.PRED_DIR + '/' + config.NAME))[2]
    if csv_file != []:
        print('*.csv file already exist in the seg folder!', flush=True)
    else:
        time.sleep(1)
        index_mean = []
        index_all = []
        index_sample = []
        zeros_cnt = 0
        mean_cnt = 0
        slice_cnt = 0
        len_data = len(dataset_test.image_ids)
        zeros_SEG = []
        for i, image_id in tqdm(enumerate(dataset_test.image_ids), total=len_data):
            # Load image and run detection
            info = dataset_test.image_info[image_id]
            path_list = info['path']
            if path_list[1] != sample_num:  # compute the mean value of one sample
                index_mean.append(np.hstack([mean_cnt, np.uint8(sample_num), np.mean(index_sample, axis=0)]))
                sample_num = path_list[1]
                mean_cnt = mean_cnt + 1
                index_sample = []
                slice_cnt = 0
            seg_path = path_list[0] + '/' + config.NAME
            gt_path = path_list[0] + '/masks'
            for single in path_list[1:]:
                gt_path = gt_path + '/' + single  # the whole path
                seg_path = seg_path + '/' + single  # the whole path
            # print('Evaluating:...\t', path_list[1] + '/' + path_list[2], '\t', image_id, '/', len_data)
            SEG = imread(seg_path)
            if np.max(SEG) == 0:
                # print("SEG is all zero!")
                zeros_SEG.append(path_list[1:])
                zeros_cnt = zeros_cnt + 1
                continue
            GT = imread(gt_path)
            GT, window, scale, padding, crop = utils.resize_image(GT, min_dim=config.IMAGE_MIN_DIM,
                                                                  min_scale=config.IMAGE_MIN_SCALE,
                                                                  max_dim=config.IMAGE_MAX_DIM,
                                                                  mode=config.IMAGE_RESIZE_MODE)
            GT = np.uint8(np.where(GT > 127, 1, 0))
            tmp_overlap_results, tmp_surface_distance_results = seg_eval_utils.evaluation_sample(SEG, GT)
            index_all.append(
                np.hstack(
                    [slice_cnt, path_list[1] + '/' + path_list[2], tmp_overlap_results,
                     tmp_surface_distance_results]))
            slice_cnt = slice_cnt + 1
            index_sample.append(np.hstack([tmp_overlap_results, tmp_surface_distance_results]))

        time.sleep(0.5)
        str1 = "Zero SEG:" + str(zeros_cnt) + "; Total Data:" + str(len_data)
        if zeros_cnt > len_data // 5:
            assert False, "Evaluation Failed! Too much zero SEG! " + str1
        else:
            print(str1, flush=True)
        print(zeros_SEG, flush=True)
        # save to csv
        # form data
        index_mean_head = ['Sample mean']
        index_result = np.hstack(['All mean', '-', np.mean(index_mean, axis=0)[2:]])
        # form header
        header1 = [name for name, _ in OverlapMeasures.__members__.items()]
        header2 = [name for name, _ in SurfaceDistanceMeasures.__members__.items()]
        header = np.hstack([header1, header2])
        header = np.hstack(['Index', 'Name', header])
        csv_name = path_list[0] + '/' + config.NAME + '/result_' + config.NAME + '.csv'
        with open(csv_name, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)  # header
            for row in index_all:  # the index of all
                writer.writerow(row)
            writer.writerow(index_mean_head)  # the header of sample mean
            for row in index_mean:  # the mean index of sample
                writer.writerow(row)
            writer.writerow(index_result)  # the mean index of all


def evaluate_folder(pred_dir, gt_dir, folder_name):
    '''
    evaluate all images in the pred_dir
    :param pred_dir:
    :param gt_dir:
    :return:
    '''
    # Define classes
    from enum import Enum
    # Use enumerations to represent the various evaluation measures
    class OverlapMeasures(Enum):
        jaccard, dice, volume_similarity, false_negative, false_positive, sensitive, specificity = range(7)

    class SurfaceDistanceMeasures(Enum):
        hausdorff_distance, mean_surface_distance, median_surface_distance, std_surface_distance, max_surface_distance = range(
            5)

    # Evaluation
    if os.path.exists(pred_dir + '/csv/' + folder_name + '.csv'):
        print(folder_name + '.csv file already exist!', flush=True)
    else:
        pred_img_paths = []
        sample_folders = next(os.walk(pred_dir + '/' + folder_name))[1]
        for sample_folder in sample_folders:
            sample_slices = next(os.walk(pred_dir + '/' + folder_name + '/' + sample_folder + '/'))[2]
            for sample_slice in sample_slices:
                pred_img_paths.append([pred_dir, sample_folder, sample_slice])

        sample_num = sample_folders[0]
        index_mean = []
        index_all = []
        index_sample = []
        zeros_cnt = 0
        mean_cnt = 0
        slice_cnt = 0
        len_data = len(pred_img_paths)
        zeros_SEG = []
        for i, pred_img_path in tqdm(enumerate(pred_img_paths), total=len_data):
            # Load image and run detection
            if pred_img_path[1] != sample_num:  # compute the mean value of one sample
                index_mean.append(np.hstack([mean_cnt, np.uint8(sample_num), np.mean(index_sample, axis=0)]))
                sample_num = pred_img_path[1]
                mean_cnt = mean_cnt + 1
                index_sample = []
                slice_cnt = 0
            seg_path = pred_dir + '/' + folder_name
            gt_path = gt_dir
            for single in pred_img_path[1:]:
                gt_path = gt_path + '/' + single  # the whole path
                seg_path = seg_path + '/' + single  # the whole path
            # print('Evaluating:...\t', path_list[1] + '/' + path_list[2], '\t', image_id, '/', len_data)
            SEG = imread(seg_path)
            SEG = np.uint8(np.where(SEG > 0, 1, 0))
            SEG_shape = list(np.shape(SEG))
            if np.max(SEG) == 0:
                # print("SEG is all zero!")
                zeros_SEG.append(pred_img_path[1:])
                zeros_cnt = zeros_cnt + 1
                continue
            GT = imread(gt_path)
            GT_shape = list(np.shape(GT))
            GT = np.where(GT > 0, 255, 0)
            if not GT_shape == SEG_shape:
                GT, window, scale, padding, crop = utils.resize_image(GT, min_dim=SEG_shape[0],
                                                                      min_scale=0,
                                                                      max_dim=SEG_shape[1],
                                                                      mode="square")
            GT = np.uint8(np.where(GT > 127, 1, 0))
            tmp_overlap_results, tmp_surface_distance_results = seg_eval_utils.evaluation_sample(SEG, GT)
            index_all.append(
                np.hstack(
                    [slice_cnt, pred_img_path[1] + '/' + pred_img_path[2], tmp_overlap_results,
                     tmp_surface_distance_results]))
            slice_cnt = slice_cnt + 1
            index_sample.append(np.hstack([tmp_overlap_results, tmp_surface_distance_results]))

        index_mean.append(np.hstack([mean_cnt, np.uint8(sample_num), np.mean(index_sample, axis=0)]))
        time.sleep(0.5)
        str1 = "Zero SEG:" + str(zeros_cnt) + "; Total Data:" + str(len_data)
        if zeros_cnt > len_data // 5:
            assert False, "Evaluation Failed! Too much zero SEG! " + str1
        else:
            print(str1, flush=True)
        print(zeros_SEG, flush=True)
        # save to csv
        # form data
        index_mean_head = ['Sample mean']
        index_result = np.hstack(['All mean', '-', np.mean(index_mean, axis=0)[2:]])
        # form header
        header1 = [name for name, _ in OverlapMeasures.__members__.items()]
        header2 = [name for name, _ in SurfaceDistanceMeasures.__members__.items()]
        header = np.hstack([header1, header2])
        header = np.hstack(['Index', 'Name', header])
        csv_name = pred_dir + '/csv/' + folder_name + '.csv'
        with open(csv_name, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)  # header
            for row in index_all:  # the index of all
                writer.writerow(row)
            writer.writerow(index_mean_head)  # the header of sample mean
            for row in index_mean:  # the mean index of sample
                writer.writerow(row)
            writer.writerow(index_result)  # the mean index of all


def evaluate_sum(dataset_test, config):
    # Define classes
    from enum import Enum
    # Use enumerations to represent the various evaluation measures
    class OverlapMeasures(Enum):
        jaccard, dice, volume_similarity, false_negative, false_positive, sensitive, specificity = range(7)

    class SurfaceDistanceMeasures(Enum):
        hausdorff_distance, mean_surface_distance, median_surface_distance, std_surface_distance, max_surface_distance = range(
            5)

    len_data = len(dataset_test.image_ids)
    info = dataset_test.image_info[0]
    path_list = info['path']
    csv_file = next(os.walk(path_list[0]))[2]
    if 'summary.csv' not in csv_file:
        folders = next(os.walk(path_list[0]))[1]
        results = []
        for folder in folders:
            csv_file = next(os.walk(path_list[0] + '/' + folder))[2]
            if csv_file != []:
                csv_file = pd.read_csv(path_list[0] + '/' + folder + '/' + csv_file[0])
                results.append(np.hstack([folder, list(csv_file.tail(1)._get_values[0][2:])]))

        # form header
        header1 = [name for name, _ in OverlapMeasures.__members__.items()]
        header2 = [name for name, _ in SurfaceDistanceMeasures.__members__.items()]
        header = np.hstack([header1, header2])
        header = np.hstack(['Index', header])
        with open(path_list[0] + '/summary.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for line in results:
                writer.writerow(line)
    else:
        print("summary.csv file already exist", flush=True)


def epoches_summary(path):
    import csv
    import pandas as pd
    # Define classes
    from enum import Enum
    # Use enumerations to represent the various evaluation measures
    class OverlapMeasures(Enum):
        jaccard, dice, volume_similarity, false_negative, false_positive, sensitive, specificity = range(7)

    class SurfaceDistanceMeasures(Enum):
        hausdorff_distance, mean_surface_distance, median_surface_distance, std_surface_distance, max_surface_distance = range(
            5)

    # form header
    header1 = [name for name, _ in OverlapMeasures.__members__.items()]
    header2 = [name for name, _ in SurfaceDistanceMeasures.__members__.items()]
    header = np.hstack([header1, header2])
    datas = np.hstack(['Index', header])
    csv_files = next(os.walk(path + '/csv'))[2]
    for csv_file in csv_files:
        data = pd.read_csv(path + '/csv/' + csv_file)
        data = list(data.tail(1)._get_values[0][2:])
        data = [csv_file] + data
        # data_epoch = epoch_folder
        # for i in range(np.shape(data)[0] - 1):
        #     data_epoch = np.vstack([data_epoch, epoch_folder])
        # if np.shape(data)[0] == 1:
        #     data = np.insert(data, 0, [data_epoch], 1)
        # else:
        #     data = np.hstack([list(data_epoch), data])
        datas = np.vstack([datas, data])
    with open(path + '/summary.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for line in datas:
            writer.writerow(line)


def enlarge_masks_save(data):
    '''
    data{'image_path','pred_path','gt_path','save_path'}
    '''
    print('Detecting data...')
    len_data = len(data['image_path'])
    pred_path = []
    for i in range(len_data):
        image_save_folder = data['save_path'][i][0] + '/image_save/' + data['save_path'][i][1]
        image_save_path = image_save_folder + '/' + data['save_path'][i][2]
        if not os.path.exists(image_save_folder):
            os.mkdir(image_save_folder)
            os.mkdir(image_save_folder)
            os.mkdir(image_save_folder)
            print('...', data['image_path'][i][-20:], '\t', i, '/', len_data)
            image_ori = imread(data['image_path'][i])
            pred = imread(data['pred_path'][i])
            gt = imread(data['gt_path'][i])
            image_enlarge, pred_enlarge, gt_enlarge = enlarge_mask(image_ori, pred, gt)
            #             self_utils.subplot_image_pred_gt(image_enlarge, pred_enlarge, gt_enlarge,0.5)

            pred_save_path = data['save_path'][i][0] + '/pred_save/' + data['save_path'][i][1] + '/' + \
                             data['save_path'][i][2]
            gt_save_path = data['save_path'][i][0] + '/gt_save/' + data['save_path'][i][1] + '/' + data['save_path'][i][
                2]
            imsave(image_save_path, image_enlarge)
            imsave(pred_save_path, pred_enlarge)
            imsave(gt_save_path, gt_enlarge)
        else:
            print('File already exists:', pred_path)


# def show_image_pred_gt(data, config):
#     '''
#     data: paths of original size images
#     data{'image_path','pred_path','gt_path'}
#     '''
#     print('Showing data...')
#     len_data = len(data['image_path'])
#     shape_pred = np.shape((data['pred_path']))
#     for i in range(len_data):
#         plt.figure(figsize=(20, 10))
#         image = imread(data['image_path'][i])
#         gt = imread(data['gt_path'][i])
#         image_enlarge, pred_enlarge, gt_enlarge = enlarge_mask(image, [], gt, config)
#         plt.subplot(1, shape_pred[0] + 1, 1)
#         plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
#         plt.axis('off')
#         plt.imshow(image_enlarge)
#         print(data['image_path'][i])
#         print(data['gt_path'][i])
#         for j in range(shape_pred[0]):
#             pred = imread(data['pred_path'][j][i])
#             print(data['pred_path'][j][i])
#             image_enlarge, pred_enlarge, gt_enlarge = enlarge_mask(image, pred, gt, config)
#             plt.subplot(1, shape_pred[0] + 1, j + 2)
#             plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
#             plt.axis('off')
#             plot_image_pred_gt(image_enlarge, pred_enlarge, gt_enlarge, 0.5)
#         plt.show()

def show_image_pred_gt(data, gap=10, figsize=(10, 10), config=None):
    '''
    data: paths of original size images
    data{'image_path','pred_path','gt_path'}
    '''
    print('Showing data...')
    len_data = len(data['image_path'])
    shape_pred = np.shape((data['pred_path']))
    for i in range(len_data):
        image = imread(data['image_path'][i])
        gt = imread(data['gt_path'][i])
        image_enlarge, pred_enlarge, gt_enlarge = enlarge_mask(image, [], gt, config)
        image_merge = image_enlarge
        image_gap = np.ones([np.shape(image_enlarge)[0], gap, 3], np.uint8) * 255
        print(data['image_path'][i])
        print(data['gt_path'][i])
        for j in range(shape_pred[0]):
            pred = imread(data['pred_path'][j][i])
            print(data['pred_path'][j][i])
            image_enlarge, pred_enlarge, gt_enlarge = enlarge_mask(image, pred, gt, config)
            image_tmp = merge_image_pred_gt(image_enlarge, pred_enlarge, gt_enlarge, 0.5)
            image_merge = np.hstack([image_merge, image_gap])
            image_merge = np.hstack([image_merge, image_tmp])
        # imsave('83_54.png',image_merge)
        plt.axis('off')
        plt.figure(figsize=figsize)
        plt.imshow(image_merge)
        plt.show()


########################
#   detect_dicoms
###########################
def detect_dicoms(INPUT_PATH, OUTPUT_PATH, sample_ids, slices):
    import nibabel as nib
    header_sample = nib.load(
        "F:/datasets/PWML_T1_data/PWML_T1_ori/masks/39.nii")  # the constant header(no need to modify)
    new_header = header_sample.header.copy()
    max_dim = 256
    mask_bg = np.zeros([max_dim, max_dim])  # mask background
    sample_ids_length = len(sample_ids)
    for i in range(sample_ids_length):
        if os.path.exists(OUTPUT_PATH + '/' + str(sample_ids[i]) + '.nii'):
            print(str(sample_ids[i]) + '.nii already exists!')
            continue
        masks = np.zeros([max_dim, max_dim, slices[0] - 1])
        folder_name = next(os.walk(INPUT_PATH + '/' + str(sample_ids[i])))[1]
        samples = next(os.walk(INPUT_PATH + '/' + str(sample_ids[i]) + '/' + folder_name[0]))[2]
        samples_length = len(samples)
        print('Processing(' + str(i + 1) + '/' + str(sample_ids_length) + '):' + str(sample_ids[i]))
        max_ids = slices[1] if samples_length > slices[1] else samples_length
        rangex = range(slices[0], max_ids)
        for cnt, ii in tqdm(enumerate(rangex), total=(len(rangex))):
            sample = dcm.read_file(INPUT_PATH + '/' + str(sample_ids[i]) + '/' + folder_name[0] + '/IM' + str(ii))
            image_ori = np.uint8(np.divide(sample.pixel_array, 32))

            # cv2 threshold processing
            image_mask_final = otsu_mask(image_ori)

            x1, x2, y1, y2 = crop_to_skull(image_ori)
            w = x2 - x1 + 1
            h = y2 - y1 + 1
            if (w < 30) | (h < 30):  # if image end early then break the loop and output the .nii
                print("early finish!")
                break
            image_slice = image_ori[y1:y2 + 1, x1:x2 + 1]
            image_expo = exposure.rescale_intensity(image_slice)

            # Detect
            r, image, mask = detect_targets(image_expo)

            # Resize mask to ori mri size
            target_num = len(r['rois'])
            if target_num > 0:
                mask_pred_ori = r['masks'][:, :, 0]  # initialize
                for iii in range(1, target_num):
                    mask_pred_ori = np.maximum(mask_pred_ori, r['masks'][:, :, iii])  # instance padding into semantic
                mask_pred_ori = np.uint8(np.ceil(mask_pred_ori))
                if h > w:
                    mask_pred = transform.resize(mask_pred_ori, (h, h), mode='reflect')
                    mask_pred = np.uint8(np.ceil(mask_pred))
                    small_pad = round((h - w) / 2)
                    top_pad = y1
                    bottom_pad = max_dim - y2 - 1
                    left_pad = x1 - small_pad
                    right_pad = max_dim - h - left_pad
                    mask_padding = [(top_pad, bottom_pad), (left_pad, right_pad)]
                    #                 print('h:',h,'w',w,'top_pad',top_pad,'bottom_pad',bottom_pad,'left_pad',left_pad,'right_pad',right_pad)
                    mask = np.pad(mask_pred, mask_padding, mode='constant', constant_values=0)

                else:
                    mask_pred = transform.resize(mask_pred_ori, (w, w), mode='reflect')
                    mask_pred = np.uint8(np.ceil(mask_pred))
                    small_pad = round((w - h) / 2)
                    top_pad = y1 - small_pad
                    bottom_pad = max_dim - w - top_pad
                    left_pad = x1
                    right_pad = max_dim - x2 - 1
                    mask_padding = [(top_pad, bottom_pad), (left_pad, right_pad)]
                    mask = np.pad(mask_pred, mask_padding, mode='constant', constant_values=0)

            else:
                mask_pred_ori = np.zeros([1024, 1024])
                mask = np.zeros([max_dim, max_dim])
            mask = np.logical_and(image_mask_final, mask)
            masks = np.concatenate((masks, np.expand_dims(mask, axis=-1)), axis=2)

        masks_shape2 = np.shape(masks)[2]
        if samples_length > masks_shape2:
            masks = np.concatenate((masks, np.zeros([max_dim, max_dim, samples_length - masks_shape2])), axis=2)
        print(samples_length, np.shape(masks))

        # Write mask data to mask.nii
        masks = masks.swapaxes(0, 1)
        mask_nii = nib.Nifti1Image(masks, None, header=new_header)
        nib.save(mask_nii, OUTPUT_PATH + '/' + str(sample_ids[i]) + '.nii')
