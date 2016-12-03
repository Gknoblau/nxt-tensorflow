import tensorflow as tf
import numpy as np
import sys
import random

dataFile = "../TF/test_data.txt"
testFile = "../TF/test.txt"
df = open(dataFile)
tFile = open(testFile)

examples = []
testFile = []

for line in df:
    data = line.split(",")
    input = data[0:4]
    output = data[4:6]
    for i in range(0, len(input)):
        input[i] = float(input[i]) / 255.0
    for i in range(0, len(output)):
        output[i] = float(output[i]) / 255.0
    examples.append((input, output))

for line in tFile:
    data = line.split(",")
    input = data[0:4]
    output = data[4:6]
    for i in range(0, len(input)):
        input[i] = float(input[i]) / 255.0
    for i in range(0, len(output)):
        output[i] = float(output[i]) / 255.0
    testFile.append((input, output))



batchSize = 32

inputX = tf.placeholder(tf.float32, (batchSize, 4))
outputY = tf.placeholder(tf.float32, (batchSize, 2))

weight = tf.get_variable("mweight", (4, 10), initializer=tf.random_normal_initializer(mean=0.1))
bias = tf.get_variable("mbias", (10, ), initializer=tf.random_normal_initializer(mean=0.1))

weight2 = tf.get_variable("mweight2", (10, 2), initializer=tf.random_normal_initializer(mean=0.1))
bias2 = tf.get_variable("mbias2", (2, ), initializer=tf.random_normal_initializer(mean=0.1))

hidden = tf.nn.relu(tf.matmul(inputX, weight) + bias)
output = tf.nn.relu(tf.matmul(hidden, weight2) + bias2)

loss = tf.reduce_sum(tf.abs(outputY - output)) # works best
#loss = tf.nn.l2_loss(outputY - output)

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
        
        # testX = []
        # testY = []
        # for i in range(0, batchSize):
        #     rX, rY = random.choice(testFile)
        #     testX.append(rX)
        #     testY.append(rY)

        # rLoss = sess.run(loss, feed_dict={inputX: testX, outputY: testY})
        # print "Test Loss {}".format(rLoss)
