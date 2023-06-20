import numpy as np

with open('logs/ssd-mobilenet-w-cache-hlo-second-time.npy', 'rb') as f:
    detection_classes_output_wo_opt = np.load(f)
    detection_boxes_output_wo_opt = np.load(f)
    detection_scores_output_wo_opt = np.load(f)
    num_detections_output_wo_opt = np.load(f)

with open('logs/ssd-mobilenet-w-cache-hlo-first-time.npy', 'rb') as f:
    detection_classes_output_wo_cache = np.load(f)
    detection_boxes_output_wo_cache = np.load(f)
    detection_scores_output_wo_cache = np.load(f)
    num_detections_output_wo_cache = np.load(f)

np.testing.assert_allclose(detection_classes_output_wo_opt, detection_classes_output_wo_cache)
np.testing.assert_allclose(detection_boxes_output_wo_opt, detection_boxes_output_wo_cache)
np.testing.assert_allclose(detection_scores_output_wo_opt, detection_scores_output_wo_cache)
np.testing.assert_allclose(num_detections_output_wo_opt, num_detections_output_wo_cache)