import tensorflow as tf
from tensorflow.python.platform import gfile

graph_def = tf.get_default_graph().as_graph_def()
with gfile.FastGFile('../saved-models/resnet_v2_fp32_savedmodel_NCHW/1538687196/saved_model.pb', 'rb') as f:
    graph_def.ParseFromString(f.read())
tf.import_graph_def(graph_def, name='')
tf.io.write_graph(graph_def, "../saved-models/resnet_v2_fp32_savedmodel_NCHW/1538687196/", "saved_model.pbtxt", as_text=True)
tf.summary.FileWriter('../events/', graph=graph_def)
