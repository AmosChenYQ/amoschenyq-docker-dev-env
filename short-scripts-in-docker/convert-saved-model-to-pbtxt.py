import re
import numpy as np
import tensorflow as tf

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



with tf.Session(graph=tf.Graph()) as sess:
  # restore sess from saved model according to tag
  tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], '../saved-models-from-zoo/resnet50_v1_cifar_saved_model/1679842215/')
  
  graph = tf.get_default_graph()
  graph_def = tf.get_default_graph().as_graph_def()
  tf.summary.FileWriter('events/', graph=graph_def)
  # print([op.name for op in graph.get_operations()])
  input_tensor = tf.get_default_graph().get_tensor_by_name("input_tensor:0")
  softmax_node = tf.get_default_graph().get_tensor_by_name("softmax_tensor:0")
  argmax_node = tf.get_default_graph().get_tensor_by_name("ArgMax:0")

  frozen_gd = tf.graph_util.convert_variables_to_constants(
            sess, tf.get_default_graph().as_graph_def(), ['ArgMax', 'softmax_tensor'])
  ensure_graph_is_valid(frozen_gd)
  tf.summary.FileWriter('events/', graph=frozen_gd)
  tf.io.write_graph(frozen_gd, "pb/", "resnet50_v1_cifar.pbtxt", as_text=True)
  tf.io.write_graph(frozen_gd, "pb/", "resnet50_v1_cifar.pb", as_text=False)

# tf.io.write_graph(graph_def, "../saved-models-from-zoo/resnet_v2_fp32_savedmodel_NCHW/1538687196/", "saved_model.pbtxt", as_text=True)