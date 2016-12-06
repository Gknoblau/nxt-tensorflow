#import nxt.locator
#from nxt.sensor import *
import tensorflow as tf
import numpy as np

batchSize = 1
inputX = tf.placeholder(tf.float32, (batchSize, 256*4))
outputY = tf.placeholder(tf.float32, (batchSize, 256*2))

weight = tf.get_variable("mweight", (256*4, 256*2), initializer=tf.random_normal_initializer(mean=0.1))

output = tf.matmul(inputX, weight)
o1 = tf.argmax(tf.nn.softmax(output[:, 0:256]), 1)
o2 = tf.argmax(tf.nn.softmax(output[:, 256:]), 1)

saver = tf.train.Saver()

def arrayToInput(ar):
    orray = np.zeros((256*4,))
    for i in range(0, 4):
        orray[i*256 + ar[i]] = 1.0
    return orray

with tf.Session() as sess:
    saver.restore(sess, "model.ckpt")
    op1, op2 = sess.run([o1, o2], feed_dict={inputX: [arrayToInput([100,100,255,255])]})
    print op1
    print op2
