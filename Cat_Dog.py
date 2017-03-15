import glob
import os
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import PIL
from PIL import Image

def norm_image(img):
    """
    Normalize PIL image

    Normalizes luminance to (mean,std)=(0,1), and applies a [1%, 99%] contrast stretch
    """
    img_y, img_b, img_r = img.convert('YCbCr').split()

    img_y_np = np.asarray(img_y).astype(float)

    img_y_np /= 255
    img_y_np -= img_y_np.mean()
    img_y_np /= img_y_np.std()
    scale = np.max([np.abs(np.percentile(img_y_np, 1.0)),
                    np.abs(np.percentile(img_y_np, 99.0))])
    img_y_np = img_y_np / scale
    img_y_np = np.clip(img_y_np, -1.0, 1.0)
    img_y_np = (img_y_np + 1.0) / 2.0

    img_y_np = (img_y_np * 255 + 0.5).astype(np.uint8)

    img_y = Image.fromarray(img_y_np)

    img_ybr = Image.merge('YCbCr', (img_y, img_b, img_r))

    img_nrm = img_ybr.convert('RGB')
    return img_nrm


def resize_image(img, size):
    """
    Resize PIL image

    Resizes image to be square with sidelength size. Pads with black if needed.
    """
    # Resize
    n_x, n_y = img.size
    if n_y > n_x:
        n_y_new = size
        n_x_new = int(size * n_x / n_y + 0.5)
    else:
        n_x_new = size
        n_y_new = int(size * n_y / n_x + 0.5)
    img_res = img.resize((n_x_new, n_y_new), resample=PIL.Image.BICUBIC)

    # Pad the borders to create a square image
    img_pad = Image.new('RGB', (size, size), (128, 128, 128))
    ulc = ((size - n_x_new) // 2, (size - n_y_new) // 2)
    img_pad.paste(img_res, ulc)
    return img_pad

#Set sample path
train_path = "train/"
test_path = "test/"

#Set Train size for small test
train_size = 2000
train_cat = train_size/2
train_dog = train_size-train_cat

valid_rate = 0.2

#Set test size
test_size = 500

image_size = 224        #set image size the same for training
depth = 3               #set the image as RGB or GreyScale

train_dog_img =[train_path + name for name in os.listdir(train_path) if 'dog' in name]
train_cat_img =[train_path + name for name in os.listdir(train_path) if 'cat' in name]
test_set_img = [test_path + name for name in os.listdir(test_path)]

train_set = train_dog_img[:train_dog] + train_cat_img[:train_cat]
test_set = test_set_img[:test_size]
train_lable = np.array ((['dogs'] * train_dog) + (['cats'] * train_dog))

# img = Image.open(train_set[1])
# # Normalize it
# img_nrm = norm_image(img)
#
# # Resize it
# img_res = resize_image(img_nrm, image_size)
# img_data = np.array(img_res, dtype=np.float32)
# img_data = (img_data - 255.0//2)/255.0
# print img_data[100]
# plt.figure(figsize=(8,4))
# plt.subplot(131)
# plt.title('Original')
# plt.imshow(img)
#
# plt.subplot(132)
# plt.title('Normalized')
# plt.imshow(img_nrm)
#
# plt.subplot(133)
# plt.title('Resized')
# plt.imshow(img_res)
#
# plt.tight_layout()
# plt.show()


def prep_data(images, size, depth):
    count = len(images)
    data = np.ndarray((count, size, size, depth), dtype=np.float32)

    for i, image_file in enumerate(images):
        image = Image.open(image_file)
        image_nrm = norm_image(image)
        image_res = resize_image(image_nrm, size)
        image_data = np.array(image_res.convert('RGB'), dtype=np.float32)
        image_data = (image_data - 255.0//2)-255.0
        data[i] = image_data;  # image_data.T
        if i % 250 == 0: print 'Processed {} of {}'.format(i, count)
    return data
#print prep_data(train_set, image_size, depth=3).shape

train_norm = prep_data(train_set, image_size, depth)
test_norm = prep_data(test_set, image_size, depth)

labels = (train_lable=='cats').astype(np.float32); # set dogs to 0 and cats to 1
labels = (np.arange(2) == labels[:,None]).astype(np.float32)

train_dataset, valid_dataset, train_labels, valid_labels = train_test_split(train_norm,
                                                                            labels,
                                                                            test_size = valid_rate,
                                                                            random_state=4)
print train_dataset.shape, valid_dataset.shape, train_labels.shape, valid_labels.shape



####build tensorflow model
import tensorflow as tf

def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, dtype=tf.float32, stddev=0.1)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape, dtype=tf.float32)
    return tf.Variable(initial, trainable=True, name=name)

def conv2d(x, W, name=name):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name=name)

def max_pool_2x2(x, name=name):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name=name)

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre= sess.run(prediction, feed_dict={train_x: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={train_x: v_xs, train_y: v_ys, keep_prob: 1})
    return result

batch_size = 32
num_labels = 2
keep_prob = tf.placeholder(tf.float32)
train_x=tf.placeholder(tf.float32,
                       shape=(None, image_size, image_size, depth))
train_y=tf.placeholder(tf.float32,
                       shape=(None, num_labels))


## conv1 layer ##
W_conv1 = weight_variable([3,3, 3,32], name='weights_conv1') # patch 3x3, in size 3, out size 32
b_conv1 = bias_variable([32], name='bias_conv1')
h_conv1 = tf.nn.relu(conv2d(train_x, W_conv1, name='conv1') + b_conv1) # output size 224x224x32
h_pool1 = max_pool_2x2(h_conv1, name='pool1')     # output size 112x112x32
final_size = image_size/2
    #
# ## conv2 layer ##
W_conv2 = weight_variable([3,3, 32, 32], name='weights_conv2') # patch 3x3, in size 32, out size 32
b_conv2 = bias_variable([32], name='bias_conv2')
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, name='conv2') + b_conv2) # output size 112x112x32
h_pool2 = max_pool_2x2(h_conv2, name='pool2')                                         # output size 56x56x32
final_size = final_size/2
# ## conv3 layer ##
W_conv3 = weight_variable([3,3, 32, 64], name='weights_conv3') # patch 3x3, in size 32, out size 64
b_conv3 = bias_variable([64], name='bias_conv3')
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3, name='conv3') + b_conv3) # output size 56x56x64
h_pool3 = max_pool_2x2(h_conv3, name='pool3')                                         # output size 28x28x64
final_size = final_size/2

## fc1 layer ##
W_fc1 = weight_variable([final_size*final_size*64, 64], name='weights_fc1')
b_fc1 = bias_variable([64], name='bias_fc1')
# [n_samples, 28, 28, 64] ->> [n_samples, 28*28*64]
h_pool3_flat = tf.reshape(h_pool3, [-1, final_size*final_size*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

## fc2 layer ##
W_fc2 = weight_variable([64, 2], name='weights_fc2')
b_fc2 = bias_variable([2], name='bias_fc2')
#fc2_l = tf.nn.bias_add(tf.matmul(h_fc1_drop, W_fc2), b_fc2)
log = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

prediction = tf.nn.softmax(log)
#
# loss = tf.reduce_mean(-tf.reduce_sum(train_y * tf.log(prediction),
#                                               reduction_indices=[1]))       # loss

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(log, train_y))
#prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)



##run
sess= tf.Session()
init=tf.global_variables_initializer()
sess.run(init)

for step in range(1000):
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    batch_x = train_dataset[offset:(offset+batch_size), :, :, :]
    batch_y = train_labels[offset:(offset+batch_size), :]
    sess.run(optimizer, feed_dict={train_x: batch_x, train_y: batch_y, keep_prob:0.5})
    if step % 50 == 0:
        print "train_accuracy: ", compute_accuracy(batch_x, batch_y)
        print "validation_accuracy: ", compute_accuracy(valid_dataset, valid_labels)
