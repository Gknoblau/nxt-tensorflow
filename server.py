from flask import Flask, request
import tensorflow as tf

batchSize = 1
inputX = tf.placeholder(tf.float32, (batchSize, 4))
weight = tf.get_variable("mweight", (4, 10), initializer=tf.random_normal_initializer(mean=0.01))
bias = tf.get_variable("mbias", (10, ), initializer=tf.random_normal_initializer(mean=0.01))

weight2 = tf.get_variable("mweight2", (10, 2), initializer=tf.random_normal_initializer(mean=0.01))
bias2 = tf.get_variable("mbias2", (2, ), initializer=tf.random_normal_initializer(mean=0.01))

hidden = tf.nn.relu(tf.matmul(inputX, weight) + bias)
output = tf.nn.relu(tf.matmul(hidden, weight2) + bias2)

saver = tf.train.Saver()

def create_app():
    app = Flask(__name__)
    app.route("/tf", methods=["post"])(handle_request)
    return app

def handle_request():
    inputs = request.get_data()
    inputs = inputs.split(' ')
    mic = inputs[0:2]
    ultra = inputs[2:4]
    for i in range(0, len(mic)):
        mic[i] = float(mic[i]) / 100.0
    for i in range(0, len(mic)):
        ultra[i] = float(ultra[i]) / 255.0
    scaled_inputs = mic + ultra

    with tf.Session() as sess:
        saver.restore(sess, "video_weights.ckpt")
        x = sess.run(output, feed_dict={inputX: [scaled_inputs]})
    
    result = x.tolist()
    motor1 = format_motor_values(result[0][0])
    motor2 = format_motor_values(result[0][1])

    print "Motor1: " + motor1
    print "Motor2: " + motor2

    return motor1 + " " + motor2

def format_motor_values(x):
    x = x * 100
    if x > 127:
        x = 127
    if x <= 0:
        x = 1
    return str(x)

server = create_app()
server.run(host='0.0.0.0', port=3000)
