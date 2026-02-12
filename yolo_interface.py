#!/usr/bin/env python3
# YOLO interface: subscribes to Gazebo camera topic and returns normalized detection center and confidence.
"""
YOLO interface implementation using user's reference code.
yolo_interface.py: Interface for YOLO object detection using Gazebo camera topic, returns normalized detection results for RL environment.
"""
import asyncio
import time
import numpy as np
import cv2
from typing import Tuple, Optional
import torch
from gz.transport13 import Node
from gz.msgs10.image_pb2 import Image
from ultralytics import YOLO
from dataclasses import dataclass


@dataclass
class DetectedTarget:
    center_px: Tuple[float, float]
    img_size: Tuple[int, int]
    t_sec: float


class RGBSubscriber:
    def __init__(self, rgb_topic: str):
        self._node = Node()
        self._rgb_image_bgr = None
        self._node.subscribe(Image, rgb_topic, self._rgb_cb)

    def _rgb_cb(self, msg: Image):
        w = int(msg.width)
        h = int(msg.height)
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        if arr.size == w * h * 3:
            img = arr.reshape((h, w, 3))
            self._rgb_image_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif arr.size == w * h * 4:
            img = arr.reshape((h, w, 4))[:, :, :3]
            self._rgb_image_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif arr.size == w * h:
            img = arr.reshape((h, w))
            self._rgb_image_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            step = getattr(msg, "step", 0)
            if step and arr.size == h * step and step % w == 0:
                ch = step // w
                if ch in (1, 3, 4):
                    full = arr.reshape((h, step))
                    row_pixels = full[:, : (w * ch)]
                    img = row_pixels.reshape((h, w, ch))
                    if ch == 1:
                        self._rgb_image_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    else:
                        self._rgb_image_bgr = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2BGR)
            else:
                self._rgb_image_bgr = None

    def latest_bgr(self) -> Optional[np.ndarray]:
        return None if self._rgb_image_bgr is None else self._rgb_image_bgr.copy()


class YOLOTracker:
    def __init__(self, weights_path: str, conf_thres: float = 0.25):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(weights_path)
        self.model.to(device)
        self.conf_thres = conf_thres

    def detect_center(self, bgr: np.ndarray) -> Optional[DetectedTarget]:
        results = self.model.predict(bgr, verbose=False)
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return None
        conf = r.boxes.conf.cpu().numpy()
        idx = int(np.argmax(conf))
        if conf[idx] < self.conf_thres:
            return None
        xyxy = r.boxes.xyxy.cpu().numpy()[idx]
        x1, y1, x2, y2 = xyxy
        cx = float((x1 + x2) * 0.5)
        cy = float((y1 + y2) * 0.5)
        h, w = bgr.shape[:2]
        return DetectedTarget(center_px=(cx, cy), img_size=(w, h), t_sec=time.monotonic())


class YOLOInterface:
    """Interface for YOLO detection system using user's reference implementation."""

    def __init__(self, weights_path: str = "model/best.pt", rgb_topic: str = "rgbd_image/image"):
        """Initialize YOLO interface with user's implementation."""
        self.rgb_sub = RGBSubscriber(rgb_topic)
        self.yolo = YOLOTracker(weights_path)
        print(f"✅ YOLOInterface initialized with weights: {weights_path}")

    async def get_detection(self) -> Tuple[float, float, float]:
        """
        Get current detection result.

        Returns:
            Tuple[float, float, float]: (cx, cy, confidence)
                - cx: center x coordinate, normalized to [0, 1] (0=left, 1=right)
                - cy: center y coordinate, normalized to [0, 1] (0=top, 1=bottom)  
                - confidence: detection confidence [0, 1] (0=no detection, 1=perfect)
        """
        # Get latest camera frame
        img = self.rgb_sub.latest_bgr()
        if img is None:
            return 0.5, 0.5, 0.0  # No image available

        # Run YOLO detection
        det = self.yolo.detect_center(img)
        if det is not None:
            (ux, uy) = det.center_px
            (w, h) = det.img_size

            # Calculate normalized center coordinates
            cx = ux / w  # Normalize to [0, 1]
            cy = uy / h  # Normalize to [0, 1]

            # Get confidence (we need to extract this from YOLO results)
            # Run YOLO again to get confidence value
            results = self.yolo.model.predict(img, verbose=False)
            r = results[0]
            if r.boxes is not None and len(r.boxes) > 0:
                conf_values = r.boxes.conf.cpu().numpy()
                max_conf = float(np.max(conf_values))
                return cx, cy, max_conf

            return cx, cy, 0.5  # Default confidence if can't extract

        return 0.5, 0.5, 0.0  # No detection

    async def get_camera_frame(self):
        """Get latest camera frame from Gazebo."""
        return self.rgb_sub.latest_bgr()
