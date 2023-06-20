import tensorflow as tf
from tensorflow.python.platform import gfile

from google.protobuf import text_format

with open('./dump-graph/after_group_2_phase_60_BuildXlaOpsPass_424749120.pbtxt') as f:
    text_graph = f.read()
graph_def = text_format.Parse(text_graph, tf.GraphDef())
tf.train.write_graph(graph_def, './dump-graph/', 'after_group_2_phase_60_BuildXlaOpsPass_424749120.pb', as_text=False)

with tf.Session() as sess:
    model_filename ='./dump-graph/after_group_2_phase_60_BuildXlaOpsPass_424749120.pb'
    with gfile.FastGFile(model_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        g_in = tf.import_graph_def(graph_def, name='')
    tf.summary.FileWriter('events/', graph=sess.graph)