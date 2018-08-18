import timeit
from cifar10 import Cifar10
from ops import *
import numpy as np
import math
import os

def Net(inputs, dropout_kept_prob, is_training):
    inputs = inputs/255.

    conv1 = Conv(inputs=inputs, num_filters=16, is_training=is_training, name="conv1")
    batch1 = batch_norm_layer(conv1, is_training=is_training,name="batch1")
    relu1 = tf.nn.relu(batch1)
    conv2 = Conv(inputs=relu1, num_filters= 16, is_training=is_training, name="conv2")
    batch2 = batch_norm_layer(conv2, is_training=is_training,name="batch2")
    relu2 = tf.nn.relu(batch2)
    conv3= Conv(inputs=relu2, num_filters=16, name="conv3", is_training=is_training)
    batch3 = batch_norm_layer(conv3, is_training=is_training,name="batch3")
    skip1 = relu1 + batch3
    relu3 = tf.nn.relu(skip1)
    pool1 = tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="pool1")
    

    conv4= Conv(inputs=pool1, num_filters=32, name="conv4", is_training=is_training)
    batch4 = batch_norm_layer(conv4, is_training=is_training,name="batch4")
    relu4 = tf.nn.relu(batch4)
    conv5 = Conv(inputs=relu4, num_filters=32, name="conv5", is_training=is_training)
    batch5 = batch_norm_layer(conv5, is_training=is_training,name="batch5")
    relu5 = tf.nn.relu(batch5)
    conv6 = Conv(inputs=relu5, num_filters=32, name="conv6", is_training=is_training)
    batch6 = batch_norm_layer(conv6, is_training=is_training,name="batch6")
    skip2 = relu4 + batch6
    relu6 = tf.nn.relu(skip2)
    pool2 = tf.nn.max_pool(relu6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="pool2")


    conv7 = Conv(inputs=pool2, num_filters=128, name="conv7", is_training=is_training)
    batch7 = batch_norm_layer(conv7, is_training=is_training,name="batch7")
    relu7 = tf.nn.relu(batch7)
    conv8 = Conv(inputs=relu7, num_filters=128, name="conv8", is_training=is_training)
    batch8 = batch_norm_layer(conv8, is_training=is_training,name="batch8")
    relu8 = tf.nn.relu(batch8)
    conv9 = Conv(inputs=relu8, num_filters=128, name="conv9", is_training=is_training)
    batch9 = batch_norm_layer(conv9, is_training=is_training,name="batch9")
    skip3 = relu7 + batch9
    relu9 = tf.nn.relu(skip3)
    pool4 = tf.nn.max_pool(relu9, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="pool4")


    shape = int(np.prod(pool4.get_shape()[1:]))
    reshaped = tf.reshape(pool4, (-1, shape))

    drop = tf.nn.dropout(reshaped, dropout_kept_prob, name='drop') 

    # fc
    with tf.variable_scope('output'):
        W = tf.get_variable('W', [drop.get_shape()[1], 10], 
                            initializer=he_normal)
        b = tf.get_variable('b', [10], initializer=tf.constant_initializer(0.0))
        out = tf.matmul(drop, W) + b
    
    return out

def train(batch_size=128, num_epochs=100):
    # Always use tf.reset_default_graph() to avoid error
    tf.reset_default_graph()
    # TODO: Write your training code here
    # - Create placeholder for inputs, training boolean, dropout keep probablity
    # - Construct your model
    # - Create loss and training op
    # - Run training
    # AS IT WILL TAKE VERY LONG ON CIFAR10 DATASET TO TRAIN
    # YOU SHOULD USE tf.train.Saver() TO SAVE YOUR MODEL AFTER TRAINING
    # AT TEST TIME, LOAD THE MODEL AND RUN TEST ON THE TEST SET

    # input Tensors
    input_x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name='x')
    input_y = tf.placeholder(tf.float32, shape=[None, 10], name='y')
    dropout_kept_prob = tf.placeholder(tf.float32, name="dropout_kept_prob")
    is_training = tf.placeholder(tf.bool)
    global_step = tf.get_variable('global_step', [],
                                  initializer=tf.constant_initializer(0),
                                  trainable=False)

    
    cifar10_train = Cifar10(batch_size=batch_size, one_hot=True, test=False, shuffle=True)

    num_train = cifar10_train.num_samples
    num_batches = math.ceil(num_train / batch_size)
  

    # ConvNet
    sess = tf.Session()
    logits = Net(input_x, dropout_kept_prob, is_training)
    # logits = ResNet(input_x, is_training, dropout_kept_prob, resnet_size=32)

    # Loss Func + Regularization Loss
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=input_y)
    loss_operation = tf.reduce_mean(cross_entropy) 

    # Exponential Learning Rate Decay
    initial_learning_rate = 0.01
    learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step,
                                           decay_steps=num_batches*num_epochs, decay_rate=0.95, staircase=True)
    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss_operation, global_step=global_step)

    # Prediction, Saver
    correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(input_y, 1))
    prediction = tf.argmax(logits, 1)
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

    saver = tf.train.Saver(max_to_keep=5)
    checkpoint_dir = os.path.curdir
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")

    # Training Loop
    init = tf.global_variables_initializer()
    sess.run(init)
    for epoch in range(num_epochs):
        # num_train = cifar10_train.num_samples
        # num_batches = math.ceil(num_train / batch_size)
        for iteration in range(num_batches):
            batch_x, batch_y = cifar10_train.get_next_batch()
            _, step, acc, l = sess.run([train_op, global_step, accuracy, loss_operation], 
                                       feed_dict={input_x:batch_x, 
                                                  input_y:batch_y, 
                                                  is_training:True, 
                                                  dropout_kept_prob:0.8})
            print("step:", step, "epoch:", epoch+1, "acc:", acc, "loss", l)

            if step%100==0:
                predicted_cifar10_test_labels = []
                cifar10_test = Cifar10(batch_size=batch_size, one_hot=False, test=True, shuffle=False)
                cifar10_test_images, cifar10_test_labels = cifar10_test._images, cifar10_test._labels
                num_test_batches = math.ceil(len(cifar10_test_images) / batch_size)
                for iteration in range(num_test_batches):
                    if iteration  != (num_test_batches-1):
                        x_test_batch = cifar10_test_images[:batch_size]
                        cifar10_test_images = cifar10_test_images[batch_size:]
                    else:
                        x_test_batch = cifar10_test_images[:len(cifar10_test_images)]
                        cifar10_test_images = cifar10_test_images[len(cifar10_test_images):]
                    pred = sess.run([prediction], feed_dict={input_x:x_test_batch, is_training:False, dropout_kept_prob:1.0})[0]
                    predicted_cifar10_test_labels += pred.tolist()
                    

                correct_predict = (cifar10_test_labels.flatten() == np.array(predicted_cifar10_test_labels).flatten()).astype(np.int32).sum()
                incorrect_predict = len(cifar10_test_labels) - correct_predict
                acc_test = float(correct_predict) / len(cifar10_test_labels)
                print("\nTesting Result:", "step:", step, "epoch:", epoch+1, "acc:", acc_test, "\n")
                max_acc = 0
                if acc_test > 0.8:
                    if acc_test > max_acc:
                        max_acc = acc_test
                        path = saver.save(sess, checkpoint_prefix, global_step=global_step)
                    print("Saved model checkpoint to {}\n".format(path))

    
    return

def test(cifar10_test_images, batch_size=128):
    # Always use tf.reset_default_graph() to avoid error
    tf.reset_default_graph()
    # - Create placeholder for inputs, training boolean, dropout keep probablity
    # - Construct your model
    # (Above 2 steps should be the same as in train function)
    # - Create label prediction tensor
    # - Run testing
    # DO NOT RUN TRAINING HERE!
    # LOAD THE MODEL AND RUN TEST ON THE TEST SET
    # input Tensors
    input_x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name='x')
    input_y = tf.placeholder(tf.float32, shape=[None, 10], name='y')
    dropout_kept_prob = tf.placeholder(tf.float32, name="dropout_kept_prob")
    is_training = tf.placeholder(tf.bool)
    global_step = tf.Variable(0, name="global_step", trainable=False)
    
    # ConvNet, optimizer, accuracy
    logits = Net(input_x, dropout_kept_prob, is_training)

    prediction = tf.argmax(logits, 1)

    checkpoint_dir = os.path.join(os.path.curdir,'model')
    saver = tf.train.Saver(max_to_keep=5)
    predicted_y_test = []
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))
        print("Model restored!")
        cifar10_test = Cifar10(batch_size=batch_size, one_hot=False, test=True, shuffle=False)
        cifar10_test_images, cifar10_test_labels = cifar10_test._images, cifar10_test._labels
        num_test_batches = math.ceil(len(cifar10_test_images) / batch_size)
        for iteration in range(num_test_batches):
            if iteration  != (num_test_batches-1):
                x_test_batch = cifar10_test_images[:batch_size]
                cifar10_test_images = cifar10_test_images[batch_size:]
            else:
                x_test_batch = cifar10_test_images[:len(cifar10_test_images)]
                cifar10_test_images = cifar10_test_images[len(cifar10_test_images):]
            pred = sess.run([prediction], feed_dict={input_x:x_test_batch, is_training:False, dropout_kept_prob:1.0})[0]
            predicted_y_test += pred.tolist()
    return np.array(predicted_y_test)

