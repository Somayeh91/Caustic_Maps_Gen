# Written with the help of https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/
#https://storage.googleapis.com/tensorflow/keras-applications/inception_v3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5
# example of calculating the frechet inception distance in Keras for cifar10
import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from numpy.random import shuffle
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.datasets.mnist import load_data
# from maps_util import resize
from skimage.transform import resize


def scale_images(images, new_shape):
    ''' This function scale an array of images to a new size:
    inputs:
        images = a set of input images in shape of (image_index, x_dimension, y_dimension, n_channels)
        new_shape = a tuple for new shape (x_dimension, y_dimension, n_channels)

    output:
        returns the images in the new shape.
    '''
    images_list = list()
    for image in images:
        # resize with nearest neighbor interpolation
        new_image = resize(image, new_shape, 0)
        # store
        images_list.append(new_image)
    return asarray(images_list)


def calculate_fid(model, images1, images2):
    ''' Calculate frechet inception distance:

        input:
            model = the InceptionV3 model (run this outside of this function: model = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3)))
            images1 = a set of input images in shape of (image_index, 299, 299, 3)
            images2 = a set of output images in shape of (image_index, 299, 299, 3)
        output:
            FID score which is a scalar.   
    '''

    # calculate activations
    act1 = model.predict(images1)
    act2 = model.predict(images2)
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=True)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=True)
    # calculate sum squared difference between means
    ssdiff = numpy.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def FID_preprocess(images):

    # convert integer to floating point values
    images = images.astype('float32')

    # resize images
    images = scale_images(images, (299,299,3))


    # pre-process images
    images = preprocess_input(images)

    return images

def read_inceptionV3():
    return InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))

    