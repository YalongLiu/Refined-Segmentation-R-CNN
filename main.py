# coding: utf-8
import os
import random
import warnings
import logging
from numpy.random import seed
from tensorflow import set_random_seed

# import configs data and model
from configs.pwml import config_maskrcnn_enhancedrpn_enlargeroi_segnet_crf as ModelConfig
import load_data.load_data_pwml as load_data
import models.model_enhancedrpn_enlargeroi_segnet_crf as modellib

# Choose a mode(inference or training)
mode = 'inference'
# mode = 'training'

if __name__ == "__main__":
    config = ModelConfig.ModelConfig()  # input config in different models

    # Set random seed
    random.seed(config.RANDOM_SEED)
    seed(config.RANDOM_SEED)
    set_random_seed(config.RANDOM_SEED)

    # config.display()
    print('Name:', config.NAME)
    print("Train/Val/Test=", config.TRAIN_VAL_TEST_RATIO)

    # Load dataset
    dataset_train = load_data.TargetDataset(config)
    dataset_train.load_samples('train', config)
    dataset_train.prepare()
    dataset_val = load_data.TargetDataset(config)
    dataset_val.load_samples('val', config)
    dataset_val.prepare()
    dataset_test = load_data.TargetDataset(config)
    dataset_test.load_samples('test', config)
    dataset_test.prepare()

    # Import libs
    from mrcnn import self_utils
    import mrcnn.config_gpu_environ

    # Train mode
    if mode == 'training':
        # Create model in training mode
        model = modellib.DefineModel(mode=mode, config=config, model_dir=os.path.join(os.getcwd(), "logs"))
        # Load weights
        if config.WEIGHTS_PATH != '':
            if not os.path.exists(config.WEIGHTS_PATH):
                logging.warning(config.WEIGHTS_PATH + " cannot be found!")
            model.load_weights(config.WEIGHTS_PATH, by_name=True, exclude=[])
        print("Training " + config.LAYERS + "...")
        config.NAME = config.NAME + '_'
        model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=config.EPOCHS,
                    layers=config.LAYERS, augment=config.AUGMENT, augmentation=False)

    # Inference mode
    if mode == 'inference':
        config.IMAGES_PER_GPU = 1
        config.BATCH_SIZE = 1
        # Create model in training mode
        model = modellib.DefineModel(mode=mode, config=config,
                                     model_dir=os.path.join(os.getcwd(), "logs"))
        # Load weights
        model.load_weights(config.INFERENCE_WEIGHTS_PATH, by_name=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self_utils.post_process(dataset_test, model, config, config.NAME+"pred_out")
