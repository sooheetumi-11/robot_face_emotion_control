#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Webcam face emotion recognition node with real-time HUD and ROS 2 publishing."""

import json
import threading
import time

import cv2
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Bool, String

from .emotion_logic import EMOTION_INFO, StableEmotion, normalize_emotion
from .face_emotion_detector import FaceEmotionDetector


class FaceEmotionNode(Node):
    """Detects facial emotions in real-time and publishes to ROS 2."""

    def __init__(self):
        super().__init__('face_emotion_node')

        # Declare parameters
        self.declare_parameter('camera_id', 0)
        self.declare_parameter('frame_width', 640)
        self.declare_parameter('frame_height', 480)
        self.declare_parameter('min_confidence', 0.60)
        self.declare_parameter('stable_frames', 3)
        self.declare_parameter('stable_ratio', 0.6)
        self.declare_parameter('show_window', True)
        self.declare_parameter('detector_backend', 'opencv')
        self.declare_parameter('emotion_topic', '/face_emotion')
        self.declare_parameter('control_state_topic', '/emotion_control_state')

        p = lambda name: self.get_parameter(name).value
        show_window = p('show_window')
        self.show_window = (
            show_window if isinstance(show_window, bool)
            else str(show_window).strip().lower() in ('1', 'true', 'yes', 'on')
        )

        # Publishers & Subscribers
        self.publisher = self.create_publisher(String, p('emotion_topic'), 10)
        self.estop_pub = self.create_publisher(Bool, '/emergency_stop', 10)
        self.state_sub = self.create_subscription(
            String, p('control_state_topic'), self._state_callback, 10
        )

        self.control_state = 'STARTING'
        self.emergency_stop_latched = False
        self.stable = StableEmotion(
            window=p('stable_frames'),
            min_confidence=p('min_confidence'),
            min_ratio=p('stable_ratio'),
        )

        # Shared detection state (protected by lock)
        self._lock = threading.Lock()
        self.last_label = None
        self.last_confidence = 0.0
        self.last_region = None
        self.current_stable_emotion = None
        self.current_stable_conf = 0.0

        # Threading for DeepFace inference
        self._pending_frame = None
        self._running = True
        self._worker_busy = False

        # FPS tracking
        self.previous_frame_time = time.monotonic()
        self.fps = 0.0

        # Initialize Camera
        camera_id = int(p('camera_id'))
        self.cap = cv2.VideoCapture(camera_id)
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(p('frame_width')))
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(p('frame_height')))
            self.get_logger().info(f'Camera {camera_id} opened successfully.')
        else:
            self.get_logger().error(f'Cannot open camera with ID {camera_id}!')

        # Initialize Face Detector
        self.detector = None
        try:
            self.detector = FaceEmotionDetector(detector_backend=p('detector_backend'))
            self.get_logger().info('FaceEmotionDetector initialized successfully.')
        except Exception as exc:
            self.get_logger().error(f'Failed to initialize FaceEmotionDetector: {exc}')

        # Start worker thread for background inference
        self.worker_thread = threading.Thread(target=self._inference_worker, daemon=True)
        self.worker_thread.start()

        # Camera & Display loop at 30 FPS
        self.timer = self.create_timer(1.0 / 30.0, self.process_frame)
        self.get_logger().info('Face Emotion Node ready. Press "e": E-Stop, "r": Release, "q": Quit.')

    def _state_callback(self, msg):
        self.control_state = msg.data
        if 'EMERGENCY_STOP' in self.control_state:
            self.emergency_stop_latched = True
        elif 'WAITING' in self.control_state or 'STOPPED' in self.control_state:
            if not self.emergency_stop_latched:
                pass

    def _publish(self, emotion=None, confidence=0.0):
        msg = String()
        msg.data = json.dumps({
            'emotion': emotion or 'no_face',
            'confidence': round(float(confidence), 4),
        })
        self.publisher.publish(msg)

    def _inference_worker(self):
        """Worker thread executing DeepFace inference asynchronously to keep UI at 30 FPS."""
        while self._running:
            frame_to_process = None
            with self._lock:
                if self._pending_frame is not None:
                    frame_to_process = self._pending_frame
                    self._pending_frame = None
                    self._worker_busy = True

            if frame_to_process is None or self.detector is None:
                time.sleep(0.01)
                continue

            try:
                raw_label, raw_conf, region = self.detector.analyze(frame_to_process)
                raw_label = normalize_emotion(raw_label)
                stable_label, stable_conf = self.stable.update(raw_label, raw_conf)

                with self._lock:
                    if raw_label and raw_conf >= self.stable.min_confidence:
                        self.last_label = raw_label
                        self.last_confidence = raw_conf
                        self.last_region = region
                    else:
                        self.last_label = None
                        self.last_confidence = 0.0
                        self.last_region = None

                    self.current_stable_emotion = stable_label
                    self.current_stable_conf = stable_conf

                self._publish(stable_label, stable_conf)

            except Exception as exc:
                self.get_logger().warning(f'Inference worker exception: {exc}')
                self.stable.reset()
                with self._lock:
                    self.last_label = None
                    self.last_confidence = 0.0
                    self.last_region = None
                    self.current_stable_emotion = None
                    self.current_stable_conf = 0.0
                self._publish()

            finally:
                with self._lock:
                    self._worker_busy = False

    def _draw_hud(self, frame):
        """Draw a clean, informative HUD with color coding and Vietnamese translations."""
        h, w = frame.shape[:2]
        overlay = frame.copy()

        # Semi-transparent top banner
        cv2.rectangle(overlay, (0, 0), (w, 55), (20, 20, 20), -1)
        # Semi-transparent bottom banner
        cv2.rectangle(overlay, (0, h - 60), (w, h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

        # Title and FPS in top banner
        cv2.putText(frame, 'TurtleBot 4 - Facial Emotion Control', (15, 25),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        fps_text = f'FPS: {self.fps:.1f}'
        cv2.putText(frame, fps_text, (w - 120, 25),
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, (100, 255, 100), 1, cv2.LINE_AA)

        with self._lock:
            label = self.last_label
            conf = self.last_confidence
            region = self.last_region
            stable_label = self.current_stable_emotion

        # Face Bounding Box & Label
        if region and label:
            rx = int(region.get('x', 0))
            ry = int(region.get('y', 0))
            rw = int(region.get('w', 0))
            rh = int(region.get('h', 0))

            rx = max(0, min(rx, w - 1))
            ry = max(0, min(ry, h - 1))
            rw = max(10, min(rw, w - rx))
            rh = max(10, min(rh, h - ry))

            info = EMOTION_INFO.get(label, {})
            box_color = info.get('color', (0, 255, 0))
            vi_name = info.get('vi', label)
            en_name = info.get('en', label).upper()
            action_desc = info.get('action_vi', '')

            # Draw face rectangle
            cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), box_color, 2)

            # Draw emotion badge above bounding box
            badge_text = f'{en_name} ({vi_name}): {conf:.0%}'
            (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            badge_y1 = max(0, ry - th - 12)
            badge_y2 = ry
            cv2.rectangle(frame, (rx, badge_y1), (rx + tw + 16, badge_y2), box_color, -1)
            cv2.putText(frame, badge_text, (rx + 8, ry - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)

            # Draw action hint directly below face box
            if action_desc:
                action_text = f'-> {action_desc}'
                cv2.putText(frame, action_text, (rx, ry + rh + 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2, cv2.LINE_AA)
        else:
            status_text = 'No face detected / looking for face...'
            cv2.putText(frame, status_text, (15, 46),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (150, 150, 150), 1, cv2.LINE_AA)

        # Bottom Bar: Robot FSM State & Controls
        state_color = (0, 255, 255) if 'RUNNING' in self.control_state else (200, 200, 200)
        if 'STOP' in self.control_state or 'NO_FACE' in self.control_state:
            state_color = (180, 180, 180)
        if self.emergency_stop_latched:
            state_color = (0, 0, 255)

        cv2.putText(frame, f'Robot State: {self.control_state}', (15, h - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, state_color, 2, cv2.LINE_AA)

        controls_hint = '[e] E-Stop | [r] Release | [q] Quit'
        cv2.putText(frame, controls_hint, (15, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

        # Emergency Stop Banner
        if self.emergency_stop_latched:
            banner_overlay = frame.copy()
            cv2.rectangle(banner_overlay, (0, h // 2 - 25), (w, h // 2 + 25), (0, 0, 180), -1)
            cv2.addWeighted(banner_overlay, 0.85, frame, 0.15, 0, frame)
            estop_msg = 'EMERGENCY STOP LATCHED - Press [r] to Release'
            (etw, eth), _ = cv2.getTextSize(estop_msg, cv2.FONT_HERSHEY_DUPLEX, 0.70, 2)
            cv2.putText(frame, estop_msg, ((w - etw) // 2, h // 2 + eth // 2),
                        cv2.FONT_HERSHEY_DUPLEX, 0.70, (255, 255, 255), 2, cv2.LINE_AA)

        return frame

    def process_frame(self):
        now = time.monotonic()
        dt = max(1e-6, now - self.previous_frame_time)
        self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt) if self.fps > 0 else (1.0 / dt)
        self.previous_frame_time = now

        if self.cap.isOpened():
            ok, frame = self.cap.read()
            if ok and frame is not None:
                # Mirror frame horizontally for natural webcam interaction
                frame = cv2.flip(frame, 1)

                # Send frame to background worker if worker is not busy
                with self._lock:
                    if not self._worker_busy and self._pending_frame is None:
                        self._pending_frame = frame.copy()

                if self.show_window:
                    try:
                        hud_frame = self._draw_hud(frame)
                        cv2.imshow('TurtleBot 4 Face Emotion Control', hud_frame)
                    except cv2.error as exc:
                        self.get_logger().error(f'OpenCV display error: {exc}')
                        self.show_window = False
            else:
                self._publish()
        else:
            self._publish()

        if self.show_window:
            try:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('e'):
                    self.emergency_stop_latched = True
                    self.estop_pub.publish(Bool(data=True))
                    self.get_logger().warn('Emergency Stop triggered by user.')
                elif key == ord('r'):
                    self.emergency_stop_latched = False
                    self.estop_pub.publish(Bool(data=False))
                    self.get_logger().info('Emergency Stop released.')
                elif key == ord('q'):
                    self.get_logger().info('Quit command received.')
                    rclpy.shutdown()
            except cv2.error as exc:
                self.get_logger().error(f'OpenCV key input error: {exc}')
                self.show_window = False

    def destroy_node(self):
        self._running = False
        try:
            try:
                self._publish()
            except Exception:
                pass
            if self.cap.isOpened():
                self.cap.release()
            if self.show_window:
                try:
                    cv2.destroyAllWindows()
                except cv2.error:
                    pass
        finally:
            super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = FaceEmotionNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
