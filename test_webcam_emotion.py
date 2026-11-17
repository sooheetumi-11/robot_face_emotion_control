#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Quick standalone test script to verify webcam & DeepFace emotion detection."""

import os
import sys
import time
import cv2

# Support running directly from this directory or workspace
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from robot_face_emotion_control.emotion_logic import EMOTION_INFO, normalize_emotion
    from robot_face_emotion_control.face_emotion_detector import FaceEmotionDetector
except ImportError:
    from emotion_logic import EMOTION_INFO, normalize_emotion
    from face_emotion_detector import FaceEmotionDetector


def main():
    print("=" * 65)
    print("  TURTLEBOT 4 - TEST NHẬN DIỆN BIỂU CẢM KHUÔN MẶT (WEBCAM)")
    print("=" * 65)
    print("Khởi tạo FaceEmotionDetector (DeepFace)...")

    try:
        detector = FaceEmotionDetector(detector_backend='opencv')
        print("-> Tải model DeepFace thành công!")
    except Exception as exc:
        print(f"[LỖI] Không thể khởi tạo FaceEmotionDetector: {exc}")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[LỖI] Không thể mở webcam /dev/video0!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("-> Đã mở webcam. Nhấn 'q' trên cửa sổ OpenCV để dừng kiểm tra.")

    prev_time = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[LỖI] Không đọc được frame từ webcam.")
            break

        frame = cv2.flip(frame, 1)
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(1e-5, now - prev_time))
        prev_time = now

        # Nhận diện biểu cảm
        emotion, conf, region = detector.analyze(frame)
        emotion = normalize_emotion(emotion)

        # Hiển thị kết quả lên frame
        h, w = frame.shape[:2]
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 130, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if emotion and region:
            info = EMOTION_INFO.get(emotion, {})
            color = info.get('color', (0, 255, 0))
            vi_name = info.get('vi', emotion)
            action = info.get('action_vi', '')

            rx, ry = int(region.get('x', 0)), int(region.get('y', 0))
            rw, rh = int(region.get('w', 0)), int(region.get('h', 0))

            cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), color, 2)
            label_text = f"{emotion.upper()} ({vi_name}): {conf:.0%}"
            cv2.putText(frame, label_text, (rx, max(25, ry - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
            cv2.putText(frame, f"-> {action}", (rx, ry + rh + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        else:
            cv2.putText(frame, "Khong tim thay khuon mat...", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 2)

        cv2.imshow("Test Webcam Face Emotion", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("-> Đã kết thúc kiểm tra webcam.")


if __name__ == '__main__':
    main()
