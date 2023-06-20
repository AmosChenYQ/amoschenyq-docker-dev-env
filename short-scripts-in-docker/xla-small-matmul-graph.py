import tensorflow as tf
import numpy as np
import time

input_node = tf.placeholder(shape=[None, 28, 28],
                                dtype=tf.float32,
                                name='input_node')

# for debug tensorboard of tf2xla bridge
# with tf.xla.experimental.jit_scope():
#     square = tf.square(input_node, name='square')
#     matmul = tf.matmul(square, square, name='matmul')
#     substract = tf.subtract(matmul, matmul, name='substract')
#     add = tf.add(substract, matmul, name='add')

with tf.device('/gpu:1'):
    square = tf.square(input_node, name='square')
    matmul = tf.matmul(square, square, name='matmul')
    substract = tf.subtract(matmul, matmul, name='substract')
    add = tf.add(substract, matmul, name='add')

# for correctness check
# with tf.device('/gpu:1'):
#     matmul = tf.matmul(input_node, input_node, name='matmul')

config=tf.compat.v1.ConfigProto()
config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
config.gpu_options.per_process_gpu_memory_fraction = 0.25

time_cost = []

base_rand_np = np.random.rand(28, 28)
rand_input_1 = np.stack([base_rand_np], axis=0)
rand_input_2 = np.stack([base_rand_np for _ in range(10)], axis=0)


tf.config.experimental.enable_tensor_float_32_execution(False)

with tf.Session(config=config) as sess:
# with tf.Session() as sess:

    for _ in range(4):
        time_start = time.time()
        result1 = sess.run(add, feed_dict={input_node: np.ones((1, 28, 28))})
        time_end = time.time()
        time_cost.append(time_end-time_start)
    for _ in range(4):
        time_start = time.time()
        result2 = sess.run(add, feed_dict={input_node: np.ones((28, 28, 28))})
        time_end = time.time()
        time_cost.append(time_end-time_start)
    # for correctness check
    # time_start = time.time()
    # result1 = sess.run(matmul, feed_dict={input_node: rand_input_1})
    # time_end = time.time()
    # time_cost.append(time_end-time_start)
    # time_start = time.time()
    # result2 = sess.run(matmul, feed_dict={input_node: rand_input_2})
    # time_end = time.time()
    # time_cost.append(time_end-time_start)

    frozen_gd = tf.graph_util.convert_variables_to_constants(
            sess, tf.get_default_graph().as_graph_def(), ['matmul'])


# tf.summary.FileWriter('../events/', graph_def=frozen_gd)
# tf.io.write_graph(tf.get_default_graph(), "../pb/", "small-network-xla.pbtxt", as_text=True)
tf.io.write_graph(frozen_gd, "./pb/", "xla-small-matmul-graph.pbtxt", as_text=True)

print("Total time cost")
print(time_cost)

# for correctness check
# result_1_np = tf.make_ndarray(tf.make_tensor_proto(result1))
# result_2_np = tf.make_ndarray(tf.make_tensor_proto(result2))
# for i in range(10):
#     print(np.testing.assert_allclose(result_1_np[0], result_2_np[i]))