# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/optimize_for_inference_lib.py
import re
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

# Helper function
def ensure_graph_is_valid(graph_def):
    node_map = {}
    for node in graph_def.node:
        if node.name not in node_map:
            node_map[node.name] = node
        else:
            raise ValueError("Duplicate node names detected for ", node.name)
    for node in graph_def.node:
        for input_name in node.input:
            input_node_name = node_name_from_input(input_name)
            if input_node_name not in node_map:
                raise ValueError("Input for ", node.name, " not found: ",
                                 input_name)

# Helper function
def node_name_from_input(node_name):
    """Strips off ports and other decorations to get the underlying node name."""
    if node_name.startswith("^"):
        node_name = node_name[1:]
    m = re.search(r"(.*):\d+$", node_name)
    if m:
        node_name = m.group(1)
    return node_name

def main():
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('../saved-models-from-zoo/ssd_mobilenet_v1_coco_2018_01_28/model.ckpt.meta')
        saver.restore(sess, '../saved-models-from-zoo/ssd_mobilenet_v1_coco_2018_01_28/model.ckpt')
        # write logs to events folder tp open in tensorboard
        tf.summary.FileWriter('events/', graph=sess.graph)
        # output_node_names are from tensorboard
        frozen_gd = tf.graph_util.convert_variables_to_constants(
            sess, tf.get_default_graph().as_graph_def(), ['detection_classes', 'detection_boxes', 'detection_scores', 'num_detections'])
        ensure_graph_is_valid(frozen_gd)
        # tf.summary.FileWriter('events/', graph=frozen_gd)
        tf.io.write_graph(frozen_gd, "pb/", "ssd_mobilenet_v1_coco.pbtxt", as_text=True)
        tf.io.write_graph(frozen_gd, "pb/", "ssd_mobilenet_v1_coco.pb", as_text=False)

if __name__ == '__main__':
    tf.reset_default_graph()
    main()
    


