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

    linearData = np.zeros((256*4,))
    linearOutputData = np.zeros((256*2,))

    for i in range(0, len(input)):
        linearData[i*256 + int(input[i])] = 1.0
    for i in range(0, len(output)):
        linearOutputData[i*256 + int(output[i])] = 1.0

    
    examples.append((linearData, linearOutputData))

# for line in tFile:
#     data = line.split(",")
#     input = data[0:4]
#     output = data[4:6]
#     for i in range(0, len(input)):
#         input[i] = float(input[i]) / 255.0
#     for i in range(0, len(output)):
#         output[i] = float(output[i]) / 255.0
#     testFile.append((input, output))



batchSize = 32

inputX = tf.placeholder(tf.float32, (batchSize, 256*4))
outputY = tf.placeholder(tf.float32, (batchSize, 256*2))

weight = tf.get_variable("mweight", (256*4, 256*2), initializer=tf.random_normal_initializer(mean=0.1))

output = tf.matmul(inputX, weight)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(tf.reshape(output, (-1, 256)), tf.reshape(outputY, (-1, 256))))

gbStep = tf.Variable(initial_value=1, dtype=tf.int64, trainable=False)

optimize = tf.train.AdamOptimizer(0.1).minimize(loss)


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
        
        gbStep.assign_add(1)
        # testX = []
        # testY = []
        # for i in range(0, batchSize):
        #     rX, rY = random.choice(testFile)
        #     testX.append(rX)
        #     testY.append(rY)

        # rLoss = sess.run(loss, feed_dict={inputX: testX, outputY: testY})
        # print "Test Loss {}".format(rLoss)
