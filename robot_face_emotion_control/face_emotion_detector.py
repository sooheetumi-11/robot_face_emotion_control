# SPDX-License-Identifier: Apache-2.0
"""DeepFace adapter for facial emotion recognition with automatic venv detection."""

import os
import sys


def _resolve_deepface():
    """Ensure DeepFace is importable, checking workspace virtual environment if needed."""
    try:
        from deepface import DeepFace
        return DeepFace
    except ImportError:
        # Check standard workspace virtual environment locations
        candidates = [
            os.path.expanduser('~/Turtlebot4/.venv-face/lib/python3.10/site-packages'),
            os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '.venv-face', 'lib', 'python3.10', 'site-packages')),
            os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '.venv-face', 'lib', 'python3.10', 'site-packages')),
        ]
        for candidate in candidates:
            if os.path.isdir(candidate) and candidate not in sys.path:
                sys.path.insert(0, candidate)
        try:
            from deepface import DeepFace
            return DeepFace
        except ImportError as exc:
            raise RuntimeError(
                'DeepFace is missing. Install requirements-face.txt in the ROS Python environment '
                'or run inside .venv-face.'
            ) from exc


class FaceEmotionDetector:
    """Run DeepFace emotion analysis on a BGR OpenCV frame."""

    def __init__(self, detector_backend='opencv'):
        self._deepface = _resolve_deepface()
        self.detector_backend = detector_backend

    def analyze(self, frame):
        """Return (emotion, confidence 0..1, face region) or (None, 0.0, None)."""
        if frame is None or frame.size == 0:
            return None, 0.0, None

        try:
            result = self._deepface.analyze(
                img_path=frame,
                actions=('emotion',),
                detector_backend=self.detector_backend,
                enforce_detection=True,
                silent=True,
            )
        except Exception as exc:
            # Handle no-face detection exception gracefully
            exc_str = str(exc).lower()
            if 'face' in exc_str or 'detect' in exc_str:
                return None, 0.0, None
            raise

        if isinstance(result, list):
            if not result:
                return None, 0.0, None
            result = max(result, key=lambda item: item.get('face_confidence', 0.0))

        # Check if a valid face was detected
        face_confidence = result.get('face_confidence', 1.0)
        if face_confidence is not None and face_confidence == 0:
            return None, 0.0, None

        scores = result.get('emotion', {})
        # Target 6 emotions required by the exercise
        scores = {
            name.lower(): float(score)
            for name, score in scores.items()
            if name.lower() in ('happy', 'sad', 'neutral', 'angry', 'surprise', 'fear')
        }
        if not scores:
            return None, 0.0, None

        label, score = max(scores.items(), key=lambda item: item[1])
        region = result.get('region')
        return label, score / 100.0, region

