import TFDB
import tensorflow as tf
import tensorflow.contrib as tfc

train, test = TFDB.dataset.image.mnist_ds('tmp/mnist_ds')

# values

N = 200
K = 4
L = 8
M = 12
P = 14
R = 12
img_size = 28
num_channels = 1
num_classes = 100
batch_size = 10


train_dataset = train.batch(batch_size)
train_iterator = train_dataset.make_initializable_iterator()
train_imgs, train_labels = train_iterator.get_next()
train_imgs = tf.reshape(train_imgs, shape=[-1, 28, 28, 1])


test_dataset = test.batch(batch_size)
test_iterator = test_dataset.make_initializable_iterator()
test_imgs, test_labels = test_iterator.get_next()
test_imgs = tf.reshape(test_imgs, shape=[-1, 28, 28, 1])



#Encoder

def conv2d(input, filter_size):
    layer = tf.layers.conv2d(input, filter_size, kernel_size=(3,3), strides=(1,1), padding='SAME', use_bias=True)
    return layer

def de_conv2d(input, filter_size):
    layer = tf.layers.conv2d_transpose(input, filter_size, kernel_size=(3,3), strides=(1, 1), padding='SAME', use_bias=True)
    return layer

def max_pool(input):
    layer = tf.layers.max_pooling2d(input, pool_size=(2,2), strides=(2,2), padding='SAME')
    return layer

def model(img):
    with tf.variable_scope('model', reuse=tf.AUTO_REUSE):

        #convolutional
        img = tf.cast(img, tf.float32)
        # img = 28 x 28 x 1


        Y1 = tf.nn.relu(conv2d(img, 48))


        Y2 = conv2d(Y1, 48)
        Y2 = max_pool(Y2)
        Y2 = tf.layers.batch_normalization(Y2)
        Y2 = tf.nn.relu(Y2)


        Y3 = conv2d(Y2, 48)
        Y3 = max_pool(Y3)
        Y3 = tf.layers.batch_normalization(Y3)
        Y3 = tf.nn.relu(Y3)

        #deconvolutional
        Y4 = tf.keras.layers.UpSampling2D([2,2])(Y3)
        Y4 = de_conv2d(Y4, 48)
        Y4 = tf.nn.relu(Y4)

        Y5 = tf.keras.layers.UpSampling2D([2,2])(Y4)
        Y5 = de_conv2d(Y5, 48)
        Y5 = tf.nn.relu(Y5)

        Y6 = de_conv2d(Y5, 1)
        Y6 = tf.nn.relu(Y6)

        return Y6

y_pred_train = model(train_imgs)
y_pred_test = model(test_imgs)

# train loss and optimizer
loss_train = tf.losses.mean_squared_error(train_imgs, y_pred_train)
optimizer_train = tf.train.RMSPropOptimizer(0.001).minimize(loss_train)

# test loss
loss_test = tf.losses.mean_squared_error(test_imgs, y_pred_test)

# tensorboard
# dataset api

train_loss = tf.summary.scalar('metrics/loss', loss_train)
before_train_imgs = tf.summary.image('metrics/before_train_image', train_imgs)
after_train_imgs = tf.summary.image('metrics/after_train_image', y_pred_train)
stats_train = tf.summary.merge([train_loss, before_train_imgs, after_train_imgs])

test_loss = tf.summary.scalar('metrics/loss_t', loss_test)
before_test_imgs = tf.summary.image('metrics/before_test_image', test_imgs)
after_test_imgs = tf.summary.image('metrics/after_test_image', y_pred_test)
stats_test = tf.summary.merge([test_loss, before_test_imgs, after_test_imgs])


fwtrain = tf.summary.FileWriter(logdir='./training', graph=tf.get_default_graph())
fwtest = tf.summary.FileWriter(logdir='./testing', graph=tf.get_default_graph())


with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    i = 0
    j = 0
    for epoch in range(50):
        sess.run(train_iterator.initializer)
        while True:
            try:
                _, o_stats = sess.run([optimizer_train, stats_train])
                fwtrain.add_summary(o_stats, i)
                i += 1
                print()
            except tf.errors.OutOfRangeError:
                break

    sess.run(test_iterator.initializer)
    while True:
        try:
            test_stats = sess.run(stats_test)
            fwtest.add_summary(test_stats, i)
            i += 1
        except tf.errors.OutOfRangeError:
            break