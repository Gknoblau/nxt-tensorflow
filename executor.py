#import nxt.locator
#from nxt.sensor import *
import tensorflow as tf

batchSize = 1
inputX = tf.placeholder(tf.float32, (batchSize, 4))
weight = tf.get_variable("mweight", (4, 10), initializer=tf.random_normal_initializer(mean=0.01))
bias = tf.get_variable("mbias", (10, ), initializer=tf.random_normal_initializer(mean=0.01))

weight2 = tf.get_variable("mweight2", (10, 2), initializer=tf.random_normal_initializer(mean=0.01))
bias2 = tf.get_variable("mbias2", (2, ), initializer=tf.random_normal_initializer(mean=0.01))

hidden = tf.nn.relu(tf.matmul(inputX, weight) + bias)
output = tf.nn.relu(tf.matmul(hidden, weight2) + bias2)
#b = nxt.locator.find_one_brick()
saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, "model.ckpt")
    x = sess.run(output, feed_dict={inputX: [[34/255.0,38/255.0,1.0,1.0]]})
    print x
