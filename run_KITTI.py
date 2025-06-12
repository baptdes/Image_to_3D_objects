#!/usr/bin/env python3
import os
import time
import cv2
import numpy as np
import torch
import argparse
import pykitti

from detection_model import ObjectDetector
from depth_model import DepthEstimator
from bbox3d_utils import BBox3DEstimator, BirdEyeView

def main():
    parser = argparse.ArgumentParser(description="3D Object Detection on KITTI dataset only")
    parser.add_argument('--date', type=str, default= "2011_09_26", help='KITTI date')
    parser.add_argument('--drive', type=str, default = "0001", help='KITTI drive')
    parser.add_argument('--output_path', type=str, default='output.mp4', help='Path to output video file')
    parser.add_argument('--yolo_weights', type=str, default=None, help='Path to custom YOLO weights file')
    args = parser.parse_args()

    # Fixed KITTI base directory
    kitti_basedir = "KITTI_dataset"
    yolo_weights = "KITTI_dataset/best.pt"

    # Model/config
    yolo_model_size = "nano"
    depth_model_size = "small"
    device = 'cuda'
    conf_threshold = 0.25
    iou_threshold = 0.45
    # Only detect classes 0-6 (Car, Van, Truck, Pedestrian, Person_sitting, Cyclist, Tram) (Not the wtf classes 'misc' and 'dontcare')
    classes = [0, 1, 2, 3, 4, 5, 6]
    enable_tracking = True

    print(f"Using device: {device}")
    print(f"Loading KITTI sequence: {kitti_basedir}, {args.date}, {args.drive}")
    dataset = pykitti.raw(kitti_basedir, args.date, args.drive)
    kitti_frames = list(dataset.cam2)
    total_frames = len(kitti_frames)
    width, height = kitti_frames[0].size
    width, height = int(width), int(height)
    fps = 10

    print("Initializing models...")
    try:
        detector = ObjectDetector(
            model_size=yolo_model_size,
            conf_thres=conf_threshold,
            iou_thres=iou_threshold,
            classes=classes,
            device=device,
            model_path= yolo_weights
        )
    except Exception as e:
        print(f"Error initializing object detector: {e}")
        detector = ObjectDetector(
            model_size=yolo_model_size,
            conf_thres=conf_threshold,
            iou_thres=iou_threshold,
            classes=classes,
            device='cpu',
            model_path= yolo_weights
        )

    try:
        depth_estimator = DepthEstimator(
            model_size=depth_model_size,
            device=device
        )
    except Exception as e:
        print(f"Error initializing depth estimator: {e}")
        depth_estimator = DepthEstimator(
            model_size=depth_model_size,
            device='cpu'
        )

    bbox3d_estimator = BBox3DEstimator()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output_path, fourcc, fps, (width, height))

    frame_count = 0

    print("Starting processing...")

    while frame_count < total_frames:
        key = cv2.waitKey(1)
        if key == ord('q') or key == 27 or (key & 0xFF) == ord('q') or (key & 0xFF) == 27:
            print("Exiting program...")
            break

        try:
            pil_img = kitti_frames[frame_count]
            frame = np.array(pil_img)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            original_frame = frame.copy()
            detection_frame = frame.copy()
            result_frame = frame.copy()

            # First : 2D detection with YOLO
            try:
                detection_frame, detections = detector.detect(detection_frame, track=enable_tracking)
            except Exception as e:
                print(f"Error during object detection: {e}")
                detections = []
                cv2.putText(detection_frame, "Detection Error", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Second : Depth estimation
            try:
                depth_map = depth_estimator.estimate_depth(original_frame)
                depth_colored = depth_estimator.colorize_depth(depth_map)
            except Exception as e:
                print(f"Error during depth estimation: {e}")
                depth_map = np.zeros((height, width), dtype=np.float32)
                depth_colored = np.zeros((height, width, 3), dtype=np.uint8)
                cv2.putText(depth_colored, "Depth Error", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Third : 3D bounding box estimation
            boxes_3d = []
            active_ids = []
            # For each 2D detection
            for detection in detections:
                try:
                    # Gathering information from detection
                    bbox, score, class_id, obj_id = detection
                    class_name = detector.get_class_names()[class_id]
                    depth_value = depth_estimator.get_depth_in_region(depth_map, bbox, method='median')
                    depth_method = 'median'
                    box_3d = {
                        'bbox_2d': bbox,
                        'depth_value': depth_value,
                        'depth_method': depth_method,
                        'class_name': class_name,
                        'object_id': obj_id,
                        'score': score
                    }
                    boxes_3d.append(box_3d)
                    if obj_id is not None:
                        active_ids.append(obj_id)
                except Exception as e:
                    print(f"Error processing detection: {e}")
                    continue

            bbox3d_estimator.cleanup_trackers(active_ids)

            for box_3d in boxes_3d:
                try:
                    class_name = box_3d['class_name'].lower()
                    if 'car' in class_name:
                        color = (0, 0, 255)
                    elif 'person' in class_name:
                        color = (0, 255, 0)
                    else:
                        color = (255, 255, 255)
                    result_frame = bbox3d_estimator.draw_box_3d(result_frame, box_3d, color=color)
                except Exception as e:
                    print(f"Error drawing box: {e}")
                    continue

            cv2.putText(result_frame, f"Device: {device}", (10, 30), 
           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            out.write(result_frame)
            cv2.imshow("3D Object Detection", result_frame)

            frame_count += 1

            key = cv2.waitKey(1)
            if key == ord('q') or key == 27 or (key & 0xFF) == ord('q') or (key & 0xFF) == 27: # Exit on 'q' or 'Esc'
                print("Exiting program...")
                break

        except Exception as e:
            print(f"Error processing frame: {e}")
            key = cv2.waitKey(1)
            if key == ord('q') or key == 27 or (key & 0xFF) == ord('q') or (key & 0xFF) == 27: # Exit on 'q' or 'Esc'
                print("Exiting program...")
                break
            continue

    print("Cleaning up resources...")
    out.release()
    cv2.destroyAllWindows()
    print(f"Processing complete. Output saved to {args.output_path}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user (Ctrl+C)")
        cv2.destroyAllWindows()