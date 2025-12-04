import argparse
import cv2
import numpy as np
from impl.losses import weighted_loss
from tensorflow.keras.models import load_model
from ultralytics import YOLO
import time
import glob
import os
import tqdm

from deeparuco_processor import DeepArucoProcessor, render_results

def project_corners(corners, source_boxes):
    
    corners = corners.reshape(-1, 4, 2)

    widths = source_boxes[:, 2] - source_boxes[:, 0]
    heights = source_boxes[:, 3] - source_boxes[:, 1]
    origins = source_boxes[:, :2]

    corners[:, :, 0] = corners[:, :, 0] * widths[:,None]
    corners[:, :, 1] = corners[:, :, 1] * heights[:,None]

    corners = corners + origins[:, None, :]

    return corners


def main(data_dir: str, detector_path: str, regressor_path: str, render: bool, decoder_path="models/dec_new.h5"):

    detector = YOLO(detector_path)
    regressor = load_model(regressor_path, custom_objects={"weighted_loss": weighted_loss},
    )
    decoder = load_model(decoder_path)

    pipe = DeepArucoProcessor(detector, regressor, decoder)

    image_paths = glob.glob(os.path.join(data_dir, "*.png"))
    print(f"processing {len(image_paths)} files!")

    timings = []

    for iidx in tqdm.tqdm(range(len(image_paths))):
        
        img = cv2.imread(image_paths[iidx])
        
        t0 = time.time()
        results = pipe.infer(img)
        timings.append(time.time() - t0)

        boxes = results['bboxes']
        ext_corners = project_corners(results['corners'], results['filtered_boxes'])
        marker_ids = np.array(results['decode_ids'])
        marker_dists = np.array(results['decode_dists'])

        img = render_results(img, boxes=boxes, corners=ext_corners, marker_ids=marker_ids)   

        if render:                   
            cv2.imshow('results', img)
            cv2.setWindowTitle('results', f"results_{iidx}")
            k = cv2.waitKey(1)
            if k == ord('q'):
                break

    if render:
        cv2.destroyAllWindows()
    
    print(f"{len(timings)} files processed!")
    print(f"mean: {1000*np.mean(timings[10:]):.2f} ms, std: {1000*np.std(timings[10:]):.2f} ms")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="DeepArUco v2 demo tool.")
    parser.add_argument("--data", type=str, required=True, help="Path to directory of images.")
    parser.add_argument("--detector", type=str, help="marker detector to use", default="models/det_luma_bc_s.pt")
    parser.add_argument("--regressor", type=str, help="corner refinement model to use", default="models/reg_hmap_8.h5")
    parser.add_argument('--render', action='store_true', help='To render.')
    args = parser.parse_args()

    main(args.data, args.detector, args.regressor, args.render)
