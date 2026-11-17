# SPDX-License-Identifier: Apache-2.0
"""ROS-independent emotion definitions, Vietnamese labels, and debounce filtering."""

from collections import Counter, deque

EMOTIONS = ('happy', 'sad', 'neutral', 'angry', 'surprise', 'fear')

# Mapping for Vietnamese labels and description
EMOTION_INFO = {
    'happy': {
        'vi': 'Vui ve',
        'en': 'Happy',
        'action_vi': 'Re trai (Quy dao cong)',
        'action_en': 'Curve Left',
        'color': (50, 205, 50),     # Lime green (BGR)
    },
    'sad': {
        'vi': 'Buon',
        'en': 'Sad',
        'action_vi': 'Re phai (Quy dao cong)',
        'action_en': 'Curve Right',
        'color': (255, 140, 0),     # Deep sky blue / orange-blue (BGR)
    },
    'neutral': {
        'vi': 'Binh thuong',
        'en': 'Neutral',
        'action_vi': 'Dung tai cho',
        'action_en': 'Stop in place',
        'color': (180, 180, 180),   # Light gray (BGR)
    },
    'angry': {
        'vi': 'Tuc gian',
        'en': 'Angry',
        'action_vi': 'Lui nhe ve phia sau',
        'action_en': 'Move backward',
        'color': (0, 0, 230),       # Bright red (BGR)
    },
    'surprise': {
        'vi': 'Ngac nhien',
        'en': 'Surprise',
        'action_vi': 'Xoay mot vong 360 do',
        'action_en': 'Spin 360 deg in place',
        'color': (0, 215, 255),     # Gold / Amber (BGR)
    },
    'fear': {
        'vi': 'So hai',
        'en': 'Fear',
        'action_vi': 'Tien nhe ve phia truoc',
        'action_en': 'Move forward',
        'color': (200, 50, 200),    # Magenta / Purple (BGR)
    },
}


def normalize_emotion(name):
    """Return a supported lowercase emotion, or None for unknown/no face."""
    value = str(name or '').strip().lower()
    return value if value in EMOTIONS else None


class StableEmotion:
    """Accept an emotion after a consistent rolling window of confident frames."""

    def __init__(self, window=3, min_confidence=0.60, min_ratio=0.6):
        self.window = max(1, int(window))
        self.min_confidence = float(min_confidence)
        self.min_ratio = float(min_ratio)
        self.samples = deque(maxlen=self.window)

    def reset(self):
        self.samples.clear()

    def update(self, emotion, confidence):
        emotion = normalize_emotion(emotion)
        confidence = float(confidence)

        if emotion is not None and confidence >= self.min_confidence:
            self.samples.append((emotion, confidence))
        else:
            self.samples.append((None, 0.0))

        valid_samples = [item for item in self.samples if item[0] is not None]
        min_required = max(2, int(self.window * self.min_ratio)) if self.window >= 2 else 1
        if len(self.samples) < min_required or len(valid_samples) < min_required:
            return None, 0.0

        counts = Counter(item[0] for item in valid_samples)
        winner, count = counts.most_common(1)[0]

        # Check ratio against total window size
        if count / len(self.samples) < self.min_ratio:
            return None, 0.0

        confidences = [score for name, score in valid_samples if name == winner]
        return winner, sum(confidences) / len(confidences)

