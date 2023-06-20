import numpy as np

import tensorflow as tf
from tensorflow.python.platform import gfile
import time

from absl import app
from absl import flags

FLAGS = flags.FLAGS

# Flag names are globally defined!  So in general, we need to be
# careful to pick names that are unlikely to be used by other libraries.
# If there is a conflict, we'll get an error at import time.

BATCH_SIZE_LIST = [1, 2, 4, 8, 16, 32]
REPEAT_TIMES_PER_BATCH = 4

def inference_resnet(enable_xla):
    graph_def = tf.get_default_graph().as_graph_def()
    # with gfile.FastGFile('./pb/resnet50_v1_cifar.pb', 'rb') as f:
    with gfile.GFile('./pb/resnet50_v1_cifar.pb', 'rb') as f:
        graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')
    # graph = tf.get_default_graph()
    # print([op.name for op in graph.get_operations()])

    config=tf.compat.v1.ConfigProto()
    # if enable_xla:
    #   config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    with tf.Session(config=config) as sess:
    # with tf.Session(config=config) as sess:
        input_node = tf.get_default_graph().get_tensor_by_name('input_tensor:0')
        softmax_node = tf.get_default_graph().get_tensor_by_name('softmax_tensor:0')
        argmax_node = tf.get_default_graph().get_tensor_by_name('ArgMax:0')
        cost_time = []
        for bs in BATCH_SIZE_LIST:
          fake_input = np.random.rand(bs, 32, 32, 3)
          for repeat in range(REPEAT_TIMES_PER_BATCH):
            time_start = time.time()
            softmax_node_output, argmax_node_output = sess.run([softmax_node, argmax_node], feed_dict={input_node: fake_input})
            time_end = time.time()
            cost_time.append(1000 * (time_end - time_start))
        
    return cost_time



def inference_ssd(enable_xla, gpu_id, result_log_prefix):
    graph_def = tf.get_default_graph().as_graph_def()
    # with gfile.FastGFile('./pb/ssd_mobilenet_v1_coco.pb', 'rb') as f:
    with gfile.GFile('./pb/ssd_mobilenet_v1_coco.pb', 'rb') as f:
        graph_def.ParseFromString(f.read())

    with open('ssd-input.npy', 'rb') as f:
      loaded_input = np.load(f)
    
    # Import graph and reset name to null(default) to get tensor by name

    tf.import_graph_def(graph_def, name='')
    # graph = tf.get_default_graph()
    # print([op.name for op in graph.get_operations()])

    config = tf.compat.v1.ConfigProto()
    if enable_xla:
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    if gpu_id > 0:
        config.gpu_options.visible_device_list = str(gpu_id)
    with tf.Session(config=config) as sess:
        input_node = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
        detection_classes = tf.get_default_graph().get_tensor_by_name('detection_classes:0')
        detection_boxes = tf.get_default_graph().get_tensor_by_name('detection_boxes:0')
        detection_scores = tf.get_default_graph().get_tensor_by_name('detection_scores:0')
        num_detections = tf.get_default_graph().get_tensor_by_name('num_detections:0')
        cost_time = []
        for bs in BATCH_SIZE_LIST:
          # generate fake input from loaded input
          # fake_input = np.random.rand(bs, 224, 224, 3)
          input_arr = [loaded_input for _ in range(bs)]
          fake_input = np.stack(input_arr, axis=0)
          for repeat in range(REPEAT_TIMES_PER_BATCH):
            time_start = time.time()
            detection_classes_output, detection_boxes_output, \
            detection_scores_output, num_detections_output \
                = sess.run([detection_classes, detection_boxes, detection_scores, num_detections], feed_dict={input_node: fake_input})
            time_end = time.time()
            cost_time.append(1000 * (time_end - time_start))
            # only log when batch=1 and repeat=REPEAT_TIMES_PER_BATCH-1(last one) and log_dir is set
            if bs == 1 and repeat == 1 and result_log_prefix:
               with open(result_log_prefix + '.npy', 'wb') as f:
                  np.save(f, detection_classes_output)
                  np.save(f, detection_boxes_output)
                  np.save(f, detection_scores_output)
                  np.save(f, num_detections_output)
        
    return cost_time


def main(argv):
    cost_time = inference_ssd(FLAGS.enable_xla, FLAGS.gpu_id, FLAGS.result_log_prefix)
    print(cost_time)

if __name__ == '__main__':
    flags.DEFINE_boolean(name='enable_xla',
                         default=False,
                         help='Set to True to enable XLA')
    flags.DEFINE_integer(name='gpu_id',
                         default=0,
                         help='Set which gpu id to use')
    flags.DEFINE_string(name='result_log_prefix',
                        default='',
                        help='Set result log filename prefix to verify correctness')
    
    app.run(main)
