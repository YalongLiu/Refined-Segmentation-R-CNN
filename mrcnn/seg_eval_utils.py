import os
import numpy as np
import SimpleITK as sitk
from enum import Enum
from skimage.io import imread


# Use enumerations to represent the various evaluation measures
class OverlapMeasures(Enum):
    jaccard, dice, volume_similarity, false_negative, false_positive, sensitive, specificity = range(7)


class SurfaceDistanceMeasures(Enum):
    hausdorff_distance, mean_surface_distance, median_surface_distance, std_surface_distance, max_surface_distance = range(
        5)


def specificity(SEG, GT):
    TN = np.sum(np.logical_not(np.logical_or(SEG, GT)))
    FP = np.sum(SEG) - np.sum(np.logical_and(SEG, GT))
    spec = TN / (TN + FP)
    return spec


def evaluation_sample(SEG_np, GT_np):
    SEG = sitk.GetImageFromArray(SEG_np)
    GT = sitk.GetImageFromArray(GT_np)

    # Empty numpy arrays to hold the results
    overlap_results = np.zeros((len(OverlapMeasures.__members__.items())))
    surface_distance_results = np.zeros((len(SurfaceDistanceMeasures.__members__.items())))

    # Compute the evaluation criteria

    # Note that for the overlap measures filter, because we are dealing with a single label we
    # use the combined, all labels, evaluation measures without passing a specific label to the methods.
    overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()

    hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()

    # Use the absolute values of the distance map to compute the surface distances (distance map sign, outside or inside
    # relationship, is irrelevant)
    reference_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(SEG, squaredDistance=False))
    reference_surface = sitk.LabelContour(SEG)

    statistics_image_filter = sitk.StatisticsImageFilter()
    # Get the number of pixels in the reference surface by counting all pixels that are 1.
    statistics_image_filter.Execute(reference_surface)
    num_reference_surface_pixels = int(statistics_image_filter.GetSum())

    # Overlap measures
    overlap_measures_filter.Execute(SEG, GT)
    overlap_results[OverlapMeasures.jaccard.value] = overlap_measures_filter.GetJaccardCoefficient()
    overlap_results[OverlapMeasures.dice.value] = overlap_measures_filter.GetDiceCoefficient()
    overlap_results[OverlapMeasures.volume_similarity.value] = overlap_measures_filter.GetVolumeSimilarity()
    overlap_results[OverlapMeasures.false_negative.value] = overlap_measures_filter.GetFalseNegativeError()
    overlap_results[OverlapMeasures.false_positive.value] = overlap_measures_filter.GetFalsePositiveError()
    overlap_results[OverlapMeasures.sensitive.value] = 1 - overlap_results[OverlapMeasures.false_negative.value]
    overlap_results[OverlapMeasures.specificity.value] = specificity(SEG_np, GT_np)

    # Hausdorff distance
    hausdorff_distance_filter.Execute(SEG, GT)
    surface_distance_results[
        SurfaceDistanceMeasures.hausdorff_distance.value] = hausdorff_distance_filter.GetHausdorffDistance()
    # Symmetric surface distance measures
    segmented_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(GT, squaredDistance=False))
    segmented_surface = sitk.LabelContour(GT)

    # Multiply the binary surface SEG with the distance maps. The resulting distance
    # maps contain non-zero values only on the surface (they can also contain zero on the surface)
    seg2ref_distance_map = reference_distance_map * sitk.Cast(segmented_surface, sitk.sitkFloat32)
    ref2seg_distance_map = segmented_distance_map * sitk.Cast(reference_surface, sitk.sitkFloat32)

    # Get the number of pixels in the reference surface by counting all pixels that are 1.
    statistics_image_filter.Execute(segmented_surface)
    num_segmented_surface_pixels = int(statistics_image_filter.GetSum())

    # Get all non-zero distances and then add zero distances if required.
    seg2ref_distance_map_arr = sitk.GetArrayViewFromImage(seg2ref_distance_map)
    seg2ref_distances = list(seg2ref_distance_map_arr[seg2ref_distance_map_arr != 0])
    seg2ref_distances = seg2ref_distances + list(np.zeros(num_segmented_surface_pixels - len(seg2ref_distances)))
    ref2seg_distance_map_arr = sitk.GetArrayViewFromImage(ref2seg_distance_map)
    ref2seg_distances = list(ref2seg_distance_map_arr[ref2seg_distance_map_arr != 0])
    ref2seg_distances = ref2seg_distances + list(np.zeros(num_reference_surface_pixels - len(ref2seg_distances)))

    all_surface_distances = seg2ref_distances + ref2seg_distances

    surface_distance_results[SurfaceDistanceMeasures.mean_surface_distance.value] = np.mean(all_surface_distances)
    surface_distance_results[SurfaceDistanceMeasures.median_surface_distance.value] = np.median(
        all_surface_distances)
    surface_distance_results[SurfaceDistanceMeasures.std_surface_distance.value] = np.std(all_surface_distances)
    surface_distance_results[SurfaceDistanceMeasures.max_surface_distance.value] = np.max(all_surface_distances)

    # Print the matrices
    np.set_printoptions(precision=3)
    return overlap_results, surface_distance_results


def get_one_sample(PRED_DIR, GT_DIR):
    # get the list of slice of one sample
    gt_files_str = next(os.walk(GT_DIR))[2]
    gt_files = []
    for i in range(len(gt_files_str)):
        gt_files.append(gt_files_str[i][:-4])
    gt_files = sorted(np.uint8(gt_files))
    pred_sample = []
    gt_sample = []
    max_x = 0
    max_y = 0
    for i in range(len(gt_files)):
        # input prediction mask ans ground truth mask
        pred_mask = imread(PRED_DIR + '/' + str(gt_files[i]) + '.png')
        gt_mask = imread(GT_DIR + '/' + str(gt_files[i]) + '.png')
        if len(np.shape(pred_mask)) == 3:
            pred_mask = pred_mask[:, :, 0]
        if len(np.shape(gt_mask)) == 3:
            gt_mask = gt_mask[:, :, 0]

        # convert instance mask to semantic mask
        pred_mask = np.where(pred_mask > 0, 1, 0)
        gt_mask = np.where(gt_mask > 0, 1, 0)

        # convert uint32 to uint8
        pred_mask = np.uint8(pred_mask)
        gt_mask = np.uint8(gt_mask)

        # unify the mask shape
        y, x = np.shape(gt_mask)
        if max_x < x:
            max_x = x
        if max_y < y:
            max_y = y

        # collect slice to build voxel
        pred_sample.append(pred_mask)
        gt_sample.append(gt_mask)

    for i in range(len(gt_files)):
        y, x = np.shape(gt_sample[i])

        diff_y = max_y - y
        diff_x = max_x - x
        if diff_y > 0:
            gt_sample[i] = np.vstack([gt_sample[i], np.uint8(np.zeros([diff_y, x]))])
            pred_sample[i] = np.vstack([pred_sample[i], np.uint8(np.zeros([diff_y, x]))])
            y = max_y
        if diff_x > 0:
            gt_sample[i] = np.hstack([gt_sample[i], np.uint8(np.zeros([y, diff_x]))])
            pred_sample[i] = np.hstack([pred_sample[i], np.uint8(np.zeros([y, diff_x]))])

    pred_sample = np.swapaxes(pred_sample, 0, 2)
    SEG_np = np.swapaxes(pred_sample, 0, 1)
    gt_sample = np.swapaxes(gt_sample, 0, 2)
    GT_np = np.swapaxes(gt_sample, 0, 1)

    return SEG_np, GT_np
