# coding: utf-8
import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax

def crf_from_sigmoid(image, mask, l1=2, l2=4, c1=6, e1=1):
    '''
    Function which returns the labelled image after applying CRF
    params after debug in PWML datasets is(l1=2, l2=4, c1=6, e1=1)
    :param image: Original image
    :param mask: The output of the segmentation network
    :param l1: The size of gaussian kernel of the first location information
    :param l2: The size of gaussian kernel of the second location information
    :param c1: The size of gaussian kernel of the first color information
    :param e1: The CRF inference times
    :return: The probability mask after optimization
    '''
    image = np.ascontiguousarray(image)

    # Necessary for Bayesian Optimization debug
    l1 = round(l1)
    l2 = round(l2)
    c1 = round(c1)
    e1 = round(e1)

    # Setting up the CRF model
    d = dcrf.DenseCRF2D(image.shape[1], image.shape[0], 2)
    mask = np.expand_dims(mask, 0)
    inverse_mask = np.ones(np.shape(mask)) - mask
    full_mask = np.vstack([inverse_mask, mask])
    full_mask = np.ascontiguousarray(np.reshape(full_mask, [2, -1]))

    # get unary potentials (neg log probability)
    U = unary_from_softmax(full_mask)
    d.setUnaryEnergy(U)

    # This adds the color-independent term, features are the locations only.
    d.addPairwiseGaussian(sxy=(l1, l1), compat=3, kernel=dcrf.DIAG_KERNEL,
                          normalization=dcrf.NORMALIZE_SYMMETRIC)

    # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
    d.addPairwiseBilateral(sxy=(l2, l2), srgb=(c1, c1, c1), rgbim=image,
                           compat=10,
                           kernel=dcrf.DIAG_KERNEL,
                           normalization=dcrf.NORMALIZE_SYMMETRIC)

    # Run Inference for e1 steps
    Q = d.inference(e1)
    result = np.reshape(np.array(Q)[1], image.shape[:2])
    return result
