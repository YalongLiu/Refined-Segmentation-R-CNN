# coding: utf-8
import numpy as np
from mrcnn.config import Config

############################################################
#  Configurations
############################################################
class ModelConfig(Config):
    # The path of weights for training
    WEIGHTS_PATH = "E:\model_enhancedrpn_enlargeroi1.3_segnet_crf_pwml_98765.h5"
    # The path of weights for inference
    INFERENCE_WEIGHTS_PATH = 'E:\model_enhancedrpn_enlargeroi1.3_segnet_crf_pwml_98765.h5'

    # The Directory of data
    INPUT_DIR = "./test_data"
    IMAGE_DIR = INPUT_DIR + "/input"
    GT_DIR = INPUT_DIR + "/gt"
    PRED_DIR = INPUT_DIR + '_pred'

    # choose model and data
    MODLE_NAME = 'model_maskrcnn_enhancedrpn_enlargeroi1.3_segnet_crf'  # choose model
    DATA_NAME = 'pwml'  # choose input data
    RANDOM_SEED = 98765  # Random seed

    LAYERS = "all"  # Training layers
    LEARNING_RATE = 0.001  # lr
    EPOCHS = 100  # Epochs
    AUGMENT = True  # data simple augment
    AUGMENTATION = False  # data complex augment
    TRAIN_VAL_TEST_RATIO = [0.7, 0.15, 0.15]  # The ratio to split dataset into train/val/test

    ENLARGE_MASK = 1.3  # The coefficient of mask enlarge(N in the paper)
    enalrge_index = 1  # output_shape/input_shape of the segmentation network
    MASK_POOL_SIZE = int(round(ENLARGE_MASK * 14))  # the shape of roi_align out
    MASK_SHAPE = [enalrge_index * MASK_POOL_SIZE, enalrge_index * MASK_POOL_SIZE]  # the shape of target_mask
###################################################################################
    # The Name of the model
    NAME = MODLE_NAME + '_' + DATA_NAME + '_' + str(RANDOM_SEED) + '_'

    # Train on 1 GPU
    GPU_COUNT = 1

    # Adjust depending on your GPU memory, Batch_size = GPU_COUNT * IMAGES_PER_GPU
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 1 target

    # Number of training and validation steps per epoch
    STEPS_PER_EPOCH = 50
    VALIDATION_STEPS = 50

    # Don't exclude based on confidence. Since we have two classes
    # then 0.5 is the minimum anyway as it picks between tumor and BG
    DETECTION_MIN_CONFIDENCE = 0.7

    # Backbone network architecture
    # Supported values are: resnet50, resnet101
    BACKBONE = "resnet101"

    # Use small images for faster training. Set the limits of the small side,
    # the large side and the channel of input image that determines the image shape.
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    IMAGE_CHANNEL_COUNT = 3
    IMAGE_PADDING = True

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 100
    POST_NMS_ROIS_INFERENCE = 100

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 64

    # Image mean (RGB)
    MEAN_PIXEL = np.array([56.49, 56.49, 56.49])

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = False

    # (height, width) of the mini-mask
    MINI_MASK_SHAPE = (28, 28)

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 50

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 50

    # Max number of final detections per image
    DETECTION_MAX_INSTANCES = 50
