import tensorflow as tf
import numpy as np
import time

input_node = tf.placeholder(shape=[None, 128, 128, 3],
                                dtype=tf.float32,
                                name='input_node')
with tf.device('/gpu:1'):
    output_node = tf.layers.conv2d(input_node, 32, (3, 3), 
                                   strides=(1, 1), 
                                   padding='same', 
                                   name='conv2d')

config=tf.compat.v1.ConfigProto()
config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
config.gpu_options.per_process_gpu_memory_fraction = 0.25
print(config.graph_options.optimizer_options.global_jit_level)

time_cost = []

tf.config.experimental.enable_tensor_float_32_execution(False)

with tf.Session(config=config) as sess:
# with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    
    for _ in range(4):
        time_start = time.time()
        result1 = sess.run(output_node, feed_dict={input_node: np.ones((1, 128, 128, 3))})
        time_end = time.time()
        time_cost.append(time_end-time_start)

    for _ in range(4):
        time_start = time.time()
        result2 = sess.run(output_node, feed_dict={input_node: np.ones((28, 128, 128, 3))})
        time_end = time.time()
        time_cost.append(time_end-time_start)

    frozen_gd = tf.graph_util.convert_variables_to_constants(
            sess, tf.compat.v1.get_default_graph().as_graph_def(), ['conv2d/BiasAdd'])


# record for tensorboard
# tf.summary.FileWriter('events/', graph=tf.get_default_graph())
# tf.io.write_graph(tf.get_default_graph(), "pb/", "small-network-xla.pbtxt", as_text=True)
# write for inference or pbtxt view
tf.io.write_graph(frozen_gd, "pb/", "xla-small-conv2d-graph.pbtxt", as_text=True)

print("Total time cost")
print(time_cost)
