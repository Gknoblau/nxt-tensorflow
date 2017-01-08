import tensorflow as tf
import numpy as np
import sys
import random
import time

dataFile = "good_data_pos.txt"
df = open(dataFile)


examples = []


for line in df:
    data = line.split(",")
    mic = data[0:2]
    ultra = data[2:4]
    output = data[4:6]
    for i in range(0, len(mic)):
        mic[i] = float(mic[i]) / 100.0
    for i in range(0, len(mic)):
        ultra[i] = float(ultra[i]) / 255.0
    for i in range(0, len(output)):
        output[i] = float(output[i]) / 100.0
    examples.append((mic + ultra, output))


batchSize = 16
hiddenUnits = 10

inputX = tf.placeholder(tf.float32, (batchSize, 4))
outputY = tf.placeholder(tf.float32, (batchSize, 2))

weight = tf.get_variable("mweight", (4, hiddenUnits), initializer=tf.random_normal_initializer(mean=0.1))
bias = tf.get_variable("mbias", (hiddenUnits, ), initializer=tf.random_normal_initializer(mean=0.1))

weight2 = tf.get_variable("mweight2", (hiddenUnits, 2), initializer=tf.random_normal_initializer(mean=0.1))
bias2 = tf.get_variable("mbias2", (2, ), initializer=tf.random_normal_initializer(mean=0.1))

hidden = tf.nn.relu(tf.matmul(inputX, weight) + bias)
output = tf.nn.relu(tf.matmul(hidden, weight2) + bias2)

#loss = tf.reduce_sum(tf.abs(outputY - output)) # works okay
loss = tf.nn.l2_loss(outputY - output)

optimize = tf.train.AdamOptimizer(0.001).minimize(loss)


iterations = 100000
k=0

init = tf.initialize_all_variables()

saver = tf.train.Saver()

with tf.Session() as sess:

    sess.run(init)

    for k in range(iterations):
        batchX = []
        batchY = []
        for i in range(0, batchSize):
            rX, rY = random.choice(examples)
            batchX.append(rX)
            batchY.append(rY)

        rLoss, _ =  sess.run([loss, optimize], feed_dict={inputX: batchX, outputY: batchY})

        print "Loss was {}".format(rLoss)

        if k % 5000 == 0:
            save_path = saver.save(sess, "model.ckpt")