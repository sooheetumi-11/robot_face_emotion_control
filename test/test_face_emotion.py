# SPDX-License-Identifier: Apache-2.0
"""Automated tests for face emotion recognition and robot behavior logic."""

import json
import time
import unittest

from robot_face_emotion_control.emotion_logic import EMOTION_INFO, EMOTIONS, StableEmotion, normalize_emotion


class TestEmotionLogic(unittest.TestCase):
    def test_normalize_emotion(self):
        self.assertEqual(normalize_emotion('HAPPY'), 'happy')
        self.assertEqual(normalize_emotion('  Sad  '), 'sad')
        self.assertEqual(normalize_emotion('neutral'), 'neutral')
        self.assertEqual(normalize_emotion('ANGRY'), 'angry')
        self.assertEqual(normalize_emotion('Surprise'), 'surprise')
        self.assertEqual(normalize_emotion('fear'), 'fear')
        self.assertIsNone(normalize_emotion('disgust'))
        self.assertIsNone(normalize_emotion('unknown'))
        self.assertIsNone(normalize_emotion(''))
        self.assertIsNone(normalize_emotion(None))

    def test_emotion_info_coverage(self):
        for emo in EMOTIONS:
            self.assertIn(emo, EMOTION_INFO)
            self.assertIn('vi', EMOTION_INFO[emo])
            self.assertIn('en', EMOTION_INFO[emo])
            self.assertIn('action_vi', EMOTION_INFO[emo])
            self.assertIn('color', EMOTION_INFO[emo])

    def test_stable_emotion_debouncing(self):
        debouncer = StableEmotion(window=3, min_confidence=0.60, min_ratio=0.6)

        # Single frame not enough to meet window ratio
        lbl, conf = debouncer.update('happy', 0.85)
        self.assertIsNone(lbl)

        # Second frame meets 2/3 ratio
        lbl, conf = debouncer.update('happy', 0.90)
        self.assertEqual(lbl, 'happy')
        self.assertAlmostEqual(conf, 0.875)

        # Third frame confirms
        lbl, conf = debouncer.update('happy', 0.95)
        self.assertEqual(lbl, 'happy')

        # One noisy frame does not immediately wipe everything
        lbl, conf = debouncer.update(None, 0.0)
        self.assertEqual(lbl, 'happy')

        # Another noisy frame drops ratio below 0.6
        lbl, conf = debouncer.update(None, 0.0)
        self.assertIsNone(lbl)

        # Switch to sad
        debouncer.update('sad', 0.80)
        lbl, conf = debouncer.update('sad', 0.85)
        self.assertEqual(lbl, 'sad')


class TestBehaviorVelocityMapping(unittest.TestCase):
    def test_velocity_targets(self):
        import rclpy
        from robot_face_emotion_control.emotion_behavior_node import EmotionBehaviorNode

        rclpy.init()
        try:
            node = EmotionBehaviorNode()

            # Test Happy -> Curve Left (linear > 0, angular > 0)
            node.current_emotion = 'happy'
            node.last_message_time = time.monotonic()
            v, w = node._compute_target_velocity()
            self.assertGreater(v, 0.0, "Happy must have positive linear velocity")
            self.assertGreater(w, 0.0, "Happy must have positive angular velocity (left)")

            # Test Sad -> Curve Right (linear > 0, angular < 0)
            node.current_emotion = 'sad'
            node.last_message_time = time.monotonic()
            v, w = node._compute_target_velocity()
            self.assertGreater(v, 0.0, "Sad must have positive linear velocity")
            self.assertLess(w, 0.0, "Sad must have negative angular velocity (right)")

            # Test Neutral -> Stop (v = 0, w = 0)
            node.current_emotion = 'neutral'
            node.last_message_time = time.monotonic()
            v, w = node._compute_target_velocity()
            self.assertEqual(v, 0.0, "Neutral must stop linear motion")
            self.assertEqual(w, 0.0, "Neutral must stop angular motion")

            # Test Angry -> Move backward gently (v < 0, w = 0)
            node.current_emotion = 'angry'
            node.last_message_time = time.monotonic()
            v, w = node._compute_target_velocity()
            self.assertLess(v, 0.0, "Angry must move backward (negative linear velocity)")
            self.assertEqual(w, 0.0, "Angry must move straight backward")

            # Test Fear -> Move forward gently (v > 0, w = 0)
            node.current_emotion = 'fear'
            node.last_message_time = time.monotonic()
            v, w = node._compute_target_velocity()
            self.assertGreater(v, 0.0, "Fear must move forward (positive linear velocity)")
            self.assertEqual(w, 0.0, "Fear must move straight forward")

            # Test Surprise -> Spin 360 in place
            msg = unittest.mock.MagicMock()
            msg.data = json.dumps({'emotion': 'surprise', 'confidence': 0.85})
            node._emotion_callback(msg)
            self.assertEqual(node.active_maneuver, 'surprise')
            v, w = node._compute_target_velocity()
            self.assertEqual(v, 0.0, "Surprise must not move linearly")
            self.assertGreater(w, 0.0, "Surprise must spin in place")

            # Test Emergency Stop overrides everything
            node.emergency_stop = True
            v, w = node._compute_target_velocity()
            self.assertEqual(v, 0.0)
            self.assertEqual(w, 0.0)

            node.destroy_node()
        finally:
            rclpy.shutdown()


if __name__ == '__main__':
    import unittest.mock
    unittest.main()
