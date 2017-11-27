
# coding: utf-8

# In[ ]:


import numpy as np
import scipy.io
import scipy.misc
from scipy.misc import imresize, imread
import tensorflow as tf

###############################################################################
# Constants for the image input and output.
###############################################################################

# Output folder for the images.
OUTPUT_DIR = 'output/'
# Style image to use.
STYLE_IMAGE = 'images/the_scream.jpg'
# Content image to use.
CONTENT_IMAGE = 'images/Taipei101.jpg'
# Image dimensions constants.
IMAGE_WIDTH = 600
IMAGE_HEIGHT = 400
COLOR_CHANNELS = 3

###############################################################################
# Algorithm constants
###############################################################################
# Noise ratio. Percentage of weight of the noise for intermixing with the
# content image.
NOISE_RATIO = 0.5
# Number of iterations to run.
ITERATIONS = 500
# Constant to put more emphasis on content loss.
alpha = 1
# Constant to put more emphasis on style loss.
beta = 500
VGG_Model = 'imagenet-vgg-verydeep-19.mat'
MEAN_VALUES = np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))

CONTENT_LAYERS = [('conv4_2', 1.)]
STYLE_LAYERS = [('conv1_1', 0.2), ('conv2_1', 0.2), ('conv3_1', 0.2), ('conv4_1', 0.2), ('conv5_1', 0.2)]

def generate_noise_image(content_image, noise_ratio = NOISE_RATIO):
    """
    Returns a noise image intermixed with the content image at a certain ratio.
    """
    noise_image = np.random.uniform(
            -20, 20,
            (1, IMAGE_HEIGHT, IMAGE_WIDTH, COLOR_CHANNELS)).astype('float32')
    # White noise image from the content representation. Take a weighted average
    # of the values
    img = noise_image * noise_ratio + content_image * (1 - noise_ratio)
    return img

def load_image(path):
    image = imread(path)

    image = imresize(image, (IMAGE_HEIGHT, IMAGE_WIDTH))

    image = np.reshape(image, ((1,) + image.shape))
    # Input to the VGG net expects the mean to be subtracted.
    image = image - MEAN_VALUES
    return image

def save_image(path, image):
    # Output should add back the mean.
    image = image + MEAN_VALUES
    # Get rid of the first useless dimension, what remains is the image.
    image = image[0]
    image = np.clip(image, 0, 255).astype('uint8')
    scipy.misc.imsave(path, image)


def get_weight_bias(vgg_layers, layer_i):
    weights = vgg_layers[layer_i][0][0][2][0][0]
    w = tf.constant(weights)
    bias = vgg_layers[layer_i][0][0][2][0][1]
    b = tf.constant(np.reshape(bias, (bias.size)))
    layer_name = vgg_layers[layer_i][0][0][0]
    print layer_name
    return w, b

def conv_relu_layer(layer_input, nwb):

    conv_val = tf.nn.conv2d(layer_input, nwb[0], strides=[1, 1, 1, 1], padding='SAME')
    relu_val = tf.nn.relu(conv_val + nwb[1])

    return relu_val

def pool_layer(pool_style, layer_input):
    if pool_style == 'avg':
        return tf.nn.avg_pool(layer_input, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')
    elif pool_style == 'max':
        return  tf.nn.max_pool(layer_input, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')


def build_vgg19(path):
    net = {}
    vgg_rawnet = scipy.io.loadmat(path)
    vgg_layers = vgg_rawnet['layers'][0]
    net['input'] = tf.Variable(np.zeros((1, IMAGE_HEIGHT, IMAGE_WIDTH, 3)).astype('float32'))
    net['conv1_1'] = conv_relu_layer(net['input'], get_weight_bias(vgg_layers, 0))
    net['conv1_2'] = conv_relu_layer(net['conv1_1'], get_weight_bias(vgg_layers, 2))
    net['pool1'] = pool_layer('avg', net['conv1_2'])
    net['conv2_1'] = conv_relu_layer(net['pool1'], get_weight_bias(vgg_layers, 5))
    net['conv2_2'] = conv_relu_layer(net['conv2_1'], get_weight_bias(vgg_layers, 7))
    net['pool2'] = pool_layer('max', net['conv2_2'])
    net['conv3_1'] = conv_relu_layer(net['pool2'], get_weight_bias(vgg_layers, 10))
    net['conv3_2'] = conv_relu_layer(net['conv3_1'], get_weight_bias(vgg_layers, 12))
    net['conv3_3'] = conv_relu_layer(net['conv3_2'], get_weight_bias(vgg_layers, 14))
    net['conv3_4'] = conv_relu_layer(net['conv3_3'], get_weight_bias(vgg_layers, 16))
    net['pool3'] = pool_layer('avg', net['conv3_4'])
    net['conv4_1'] = conv_relu_layer(net['pool3'], get_weight_bias(vgg_layers, 19))
    net['conv4_2'] = conv_relu_layer(net['conv4_1'], get_weight_bias(vgg_layers, 21))
    net['conv4_3'] = conv_relu_layer(net['conv4_2'], get_weight_bias(vgg_layers, 23))
    net['conv4_4'] = conv_relu_layer(net['conv4_3'], get_weight_bias(vgg_layers, 25))
    net['pool4'] = pool_layer('max', net['conv4_4'])
    net['conv5_1'] = conv_relu_layer(net['pool4'], get_weight_bias(vgg_layers, 28))
    net['conv5_2'] = conv_relu_layer(net['conv5_1'], get_weight_bias(vgg_layers, 30))
    net['conv5_3'] = conv_relu_layer(net['conv5_2'], get_weight_bias(vgg_layers, 32))
    net['conv5_4'] = conv_relu_layer(net['conv5_3'], get_weight_bias(vgg_layers, 34))
    net['pool5'] = pool_layer('avg', net['conv5_4'])
    return net


def content_layer_loss(p, x):

    M = p.shape[1] * p.shape[2]
    N = p.shape[3]
    loss = (1. / (2 * N * M)) * tf.reduce_sum(tf.pow((x - p), 2))
    return loss


def content_loss_func(sess, net):

    layers = CONTENT_LAYERS
    total_content_loss = 0.0
    for layer_name, weight in layers:
        p = sess.run(net[layer_name])
        x = net[layer_name]
        total_content_loss += content_layer_loss(p, x)*weight

    total_content_loss /= float(len(layers))
    return total_content_loss


def gram_matrix(x, area, depth):

    x1 = tf.reshape(x, (area, depth))
    g = tf.matmul(tf.transpose(x1), x1)
    return g


def style_layer_loss(a, x):

    M = a.shape[1] * a.shape[2]
    N = a.shape[3]
    A = gram_matrix(a, M, N)
    G = gram_matrix(x, M, N)
    loss = (1. / (4 * N ** 2 * M ** 2)) * tf.reduce_sum(tf.pow((G - A), 2))

    return loss


def style_loss_func(sess, net):

    layers = STYLE_LAYERS
    total_style_loss = 0.0

    for layer_name, weight in layers:
        a = sess.run(net[layer_name])
        x = net[layer_name]
        total_style_loss += style_layer_loss(a, x) * weight

    total_style_loss /= float(len(layers))

    return total_style_loss


def main(): # 简单粗暴。
    net = build_vgg19(VGG_Model)
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    content_img = load_image(CONTENT_IMAGE)
    style_img = load_image(STYLE_IMAGE)

    sess.run([net['input'].assign(content_img)])
    cost_content = content_loss_func(sess, net)

    sess.run([net['input'].assign(style_img)])
    cost_style = style_loss_func(sess, net)

    total_loss = alpha * cost_content + beta * cost_style

    optimizer = tf.contrib.opt.ScipyOptimizerInterface(
        total_loss, method='L-BFGS-B',
        options={'maxiter': ITERATIONS,
                 'disp': 0})

    init_img = generate_noise_image(content_img)

    sess.run(tf.initialize_all_variables())
    sess.run(net['input'].assign(init_img))

    optimizer.minimize(sess)# optimize the sess

    mixed_img = sess.run(net['input']) #the sess optimized

    filename = 'output/out.png'
    save_image(filename, mixed_img)


if __name__ == '__main__':
    main()

