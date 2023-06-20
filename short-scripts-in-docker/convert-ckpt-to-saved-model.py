import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

# builder = tf.saved_model.Builder('./saved_model_demo')

builder = tf.saved_model.Builder('./resnet50_v1_cifar_saved_model')

with tf.Session() as sess:
    saver = tf.compat.v1.train.import_meta_graph('ckpt/model.ckpt-720.meta')
    saver.restore(sess, 'ckpt/model.ckpt-720')
    # saver = tf.compat.v1.train.import_meta_graph('../saved-models-from-zoo/resnet50_v1_cifar_ckpt/model.ckpt-7812.meta')
    # saver.restore(sess, '../saved-models-from-zoo/resnet50_v1_cifar_ckpt/model.ckpt-7812')
    

    graph = tf.get_default_graph()
    graph_def = tf.get_default_graph().as_graph_def()
    tf.summary.FileWriter('events/', graph=graph_def)

    logit_op = tf.compat.v1.get_default_graph().get_tensor_by_name("final_dense/MatMul:0")
    input_node = tf.compat.v1.get_default_graph().get_tensor_by_name("input_node:0")
    training_node = tf.compat.v1.get_default_graph().get_tensor_by_name("training:0")

    sig_input_tensor = tf.saved_model.build_tensor_info(input_node)
    sig_is_training_tensor = tf.saved_model.build_tensor_info(training_node)
    sig_input_tensor_signature = {tf.saved_model.CLASSIFY_INPUTS: sig_input_tensor, "is_training": sig_is_training_tensor}
    sig_output_tensor = tf.saved_model.build_tensor_info(logit_op)
    sig_output_tensor_signature = {tf.saved_model.CLASSIFY_OUTPUT_SCORES: sig_output_tensor}

    classify_signature_def = (tf.compat.v1.saved_model.build_signature_def(
                                    inputs=sig_input_tensor_signature, 
                                    outputs=sig_output_tensor_signature, 
                                    method_name=tf.saved_model.CLASSIFY_METHOD_NAME))
    
    print(classify_signature_def)
    signature_def_map = {tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY: classify_signature_def}
    builder.add_meta_graph_and_variables(sess, [tf.saved_model.SERVING], signature_def_map=signature_def_map)
    builder.save()
