# coding:utf-8
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

# config GPU
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.85
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
KTF.set_session(session)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
