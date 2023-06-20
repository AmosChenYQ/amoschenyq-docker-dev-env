import numpy as np
import tensorflow as tf
from PIL import Image

img = Image.open('./dataset/circle/2.jpg')

np_input = np.expand_dims(np.array(img), axis=0)
print(np_input.shape)

tf.reset_default_graph()
with tf.Session() as sess:
    saver = tf.train.import_meta_graph("./ckpt/model.ckpt-720.meta")
    saver.restore(sess, './ckpt/model.ckpt-720')
    # sess.run(tf.global_variables_initializer())
    # sess.run(tf.local_variables_initializer())
    logit_op = tf.get_default_graph().get_tensor_by_name("final_dense/MatMul:0")
    input_node = tf.get_default_graph().get_tensor_by_name("input_node:0")
    training_node = tf.get_default_graph().get_tensor_by_name("training:0")
    logit = sess.run(logit_op, feed_dict={input_node: np_input, training_node: False})

    # print(tf.get_default_graph())
