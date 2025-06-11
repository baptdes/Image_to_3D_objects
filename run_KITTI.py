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
    parser.add_argument('--date', type=str, required=True, help='KITTI date (e.g. 2011_09_26)')
    parser.add_argument('--drive', type=str, required=True, help='KITTI drive (e.g. 0001)')
    parser.add_argument('--output_path', type=str, default='output.mp4', help='Path to output video file')
    args = parser.parse_args()

    # Fixed KITTI base directory
    kitti_basedir = "/home/nimdaba/Documents/Image_to_3D_objects/KITTI_dataset"

    # Model/config
    yolo_model_size = "nano"
    depth_model_size = "small"
    device = 'cuda'
    conf_threshold = 0.25
    iou_threshold = 0.45
    classes = None
    enable_tracking = True
    enable_bev = True

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
            device=device
        )
    except Exception as e:
        print(f"Error initializing object detector: {e}")
        detector = ObjectDetector(
            model_size=yolo_model_size,
            conf_thres=conf_threshold,
            iou_thres=iou_threshold,
            classes=classes,
            device='cpu'
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
    if enable_bev:
        bev = BirdEyeView(scale=60, size=(300, 300))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output_path, fourcc, fps, (width, height))

    frame_count = 0
    start_time = time.time()
    fps_display = "FPS: --"

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

            try:
                detection_frame, detections = detector.detect(detection_frame, track=enable_tracking)
            except Exception as e:
                print(f"Error during object detection: {e}")
                detections = []
                cv2.putText(detection_frame, "Detection Error", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            try:
                depth_map = depth_estimator.estimate_depth(original_frame)
                depth_colored = depth_estimator.colorize_depth(depth_map)
            except Exception as e:
                print(f"Error during depth estimation: {e}")
                depth_map = np.zeros((height, width), dtype=np.float32)
                depth_colored = np.zeros((height, width, 3), dtype=np.uint8)
                cv2.putText(depth_colored, "Depth Error", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            boxes_3d = []
            active_ids = []
            for detection in detections:
                try:
                    bbox, score, class_id, obj_id = detection
                    class_name = detector.get_class_names()[class_id]
                    if class_name.lower() in ['person', 'cat', 'dog']:
                        center_x = int((bbox[0] + bbox[2]) / 2)
                        center_y = int((bbox[1] + bbox[3]) / 2)
                        depth_value = depth_estimator.get_depth_at_point(depth_map, center_x, center_y)
                        depth_method = 'center'
                    else:
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
                    if 'car' in class_name or 'vehicle' in class_name:
                        color = (0, 0, 255)
                    elif 'person' in class_name:
                        color = (0, 255, 0)
                    elif 'bicycle' in class_name or 'motorcycle' in class_name:
                        color = (255, 0, 0)
                    elif 'potted plant' in class_name or 'plant' in class_name:
                        color = (0, 255, 255)
                    else:
                        color = (255, 255, 255)
                    result_frame = bbox3d_estimator.draw_box_3d(result_frame, box_3d, color=color)
                except Exception as e:
                    print(f"Error drawing box: {e}")
                    continue

            if enable_bev:
                try:
                    bev.reset()
                    for box_3d in boxes_3d:
                        bev.draw_box(box_3d)
                    bev_image = bev.get_image()
                    bev_height = height // 4
                    bev_width = bev_height
                    if bev_height > 0 and bev_width > 0:
                        bev_resized = cv2.resize(bev_image, (bev_width, bev_height))
                        result_frame[height - bev_height:height, 0:bev_width] = bev_resized
                        cv2.rectangle(result_frame, 
                                     (0, height - bev_height), 
                                     (bev_width, height), 
                                     (255, 255, 255), 1)
                        cv2.putText(result_frame, "Bird's Eye View", 
                                   (10, height - bev_height + 20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                except Exception as e:
                    print(f"Error drawing BEV: {e}")

            frame_count += 1
            if frame_count % 10 == 0:
                end_time = time.time()
                elapsed_time = end_time - start_time
                fps_value = frame_count / elapsed_time
                fps_display = f"FPS: {fps_value:.1f}"

            cv2.putText(result_frame, f"{fps_display} | Device: {device}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            try:
                depth_height = height // 4
                depth_width = depth_height * width // height
                depth_resized = cv2.resize(depth_colored, (depth_width, depth_height))
                result_frame[0:depth_height, 0:depth_width] = depth_resized
            except Exception as e:
                print(f"Error adding depth map to result: {e}")

            out.write(result_frame)
            cv2.imshow("3D Object Detection", result_frame)
            cv2.imshow("Depth Map", depth_colored)
            cv2.imshow("Object Detection", detection_frame)

            key = cv2.waitKey(1)
            if key == ord('q') or key == 27 or (key & 0xFF) == ord('q') or (key & 0xFF) == 27:
                print("Exiting program...")
                break

        except Exception as e:
            print(f"Error processing frame: {e}")
            key = cv2.waitKey(1)
            if key == ord('q') or key == 27 or (key & 0xFF) == ord('q') or (key & 0xFF) == 27:
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