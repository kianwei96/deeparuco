import cv2
import numpy as np
import time
import tensorflow as tf
from impl.aruco import ids_as_bits, find_id_opt
from impl.heatmaps import pos_from_heatmap
from impl.losses import weighted_loss
from impl.utils import marker_from_corners, ordered_corners

import tensorflow as tf
import torch
# tf.debugging.set_log_device_placement(True)
print(f"{torch.cuda.is_available()=}")
class DeepArucoProcessor:

    def __init__(self, detector, regressor, decoder):
        self.detector = detector
        self.regressor = regressor
        self.decoder = decoder
        self.norm = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-9)
        self.ids_as_bits = torch.tensor(ids_as_bits)
        self.profiling = False

    def profile_timing(self, active: bool):
        self.profiling = active

    @tf.function(reduce_retracing=True)
    def refine_corners(self, crops):
        return self.regressor(crops)

    @tf.function(reduce_retracing=True)
    def decode_markers(self, markers):
        return self.decoder(markers)

    def infer(self, pic: np.ndarray):

        results = {'bboxes': None,
                   'filtered_boxes': None,
                   'corners': None,
                   'decode_ids': None,
                   'decode_dists': None}

        if self.profiling:
            t0 = time.time()

        #### Detect markers
        detections = self.detector(pic, verbose=False, iou=0.5, conf=0.03)[0].cpu().boxes

        if not len(detections):
            return results
        results['bboxes'] = detections.xyxy.detach().cpu().numpy()

        #### corner extraction
        xyxy = [
            [
                int(max(det[0] - (0.2 * (det[2] - det[0]) + 0.5), 0)),
                int(max(det[1] - (0.2 * (det[3] - det[1]) + 0.5), 0)),
                int(min(det[2] + (0.2 * (det[2] - det[0]) + 0.5), pic.shape[1] - 1)),
                int(min(det[3] + (0.2 * (det[3] - det[1]) + 0.5), pic.shape[0] - 1)),
            ]
            for det in [
                [int(val) for val in det.xyxy.cpu().numpy()[0]] for det in detections
            ]
        ]
        crops_ori = [
            cv2.resize(pic[det[1] : det[3], det[0] : det[2]], (64, 64)) for det in xyxy
        ]
        crops = [self.norm(crop) for crop in crops_ori]

        if self.profiling:
            print(f" BBOXDETECTION :: {1000 * (time.time() - t0):.2f} ms")    

        if self.profiling:
            t0 = time.time()

        corners = self.refine_corners(np.array(crops)).numpy()

        area = 75  # <- Expected area of the blobs to detect
        kp_params = cv2.SimpleBlobDetector_Params()
        if area > 0:
            kp_params.filterByArea = True
            kp_params.minArea = area * 0.8
            kp_params.maxArea = area * 1.2
        kp_detector = cv2.SimpleBlobDetector_create(kp_params)
        corners = [
            [(x, y) for x, y in zip(*pos_from_heatmap(pred, kp_detector))]
            for pred in corners
        ]
        keep = [len(cs) == 4 for cs in corners]
        reorg = [
                (det, crop, cs)
                for det, crop, cs, k in zip(xyxy, crops_ori, corners, keep)
                if k == True
            ]
        if not len(reorg):
            return results

        xyxy, crops_ori, corners = zip(*reorg)
        corners = [
            ordered_corners([c[0] for c in cs], [c[1] for c in cs]) for cs in corners
        ]
        results['filtered_boxes'] = np.array(xyxy)
        results['corners'] = np.array(corners)

        if self.profiling:
            print(f" REFINECORNERINFER :: {1000 * (time.time() - t0):.2f} ms")            

        if self.profiling:
            t0 = time.time() 

        #### Extract decoded ids
        markers = []
        for crop, cs in zip(crops_ori, corners):
            marker = marker_from_corners(crop, cs, 32)
            markers.append(self.norm(cv2.cvtColor(marker, cv2.COLOR_BGR2GRAY)))
        # Get ids from markers

        if self.profiling:
            print(f" DECODE (0) :: {1000 * (time.time() - t0):.2f} ms")

        decoder_out = np.round(self.decode_markers(np.array(markers)).numpy())

        if self.profiling:
            print(f" DECODE (1) :: {1000 * (time.time() - t0):.2f} ms")

        ids, dists = find_id_opt(self.ids_as_bits, torch.tensor(decoder_out).to(self.ids_as_bits.device))
        # ids, dists = zip(*[find_id(out, ids_as_bits) for out in decoder_out])

        if self.profiling:
            print(f" DECODE :: {1000 * (time.time() - t0):.2f} ms")

        results['decode_ids'] = ids
        results['decode_dists'] = dists

        return results

def render_results(img, boxes=None, corners=None, additional_corners=None, marker_ids=None, int_kps=None):

    img = img.copy()

    if boxes is not None:
        if type(boxes) is not np.ndarray:
            boxes = boxes.detach().cpu().numpy()
        for xyxy in boxes:
            cv2.rectangle(img, xyxy[:2].astype(int), xyxy[2:].astype(int), (255, 0, 0), 2)

    if corners is not None:
        if type(corners) is not np.ndarray:
            corners = corners.detach().cpu().numpy()
        for corns in corners:
            for pt in corns:
                cv2.circle(img, pt.astype(int), 4, (0, 0, 255))

    if additional_corners is not None:
        if type(additional_corners) is not np.ndarray:
            additional_corners = additional_corners.detach().cpu().numpy()
        for corns in additional_corners:
            for pt in corns:
                cv2.circle(img, pt.astype(int), 2, (0, 255, 0), -1)

    if (marker_ids is not None) and (corners is not None):
        if type(marker_ids) is not np.ndarray:
            marker_ids = marker_ids.detach().cpu().numpy()
        for mid, corns in zip(marker_ids, corners):
            cv2.putText(img, str(mid), corns.mean(0).astype(int), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    if int_kps is not None:
        if type(int_kps) is not np.ndarray:
            int_kps = int_kps.detach().cpu().numpy()
        for pt in int_kps.reshape(-1, 2):
            cv2.circle(img, pt.astype(int), 1, (0, 255, 0), -1)   

    return img  