#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""TurtleBot 4 behavior node controlling trajectories from facial emotions."""

import json
import math
import time

from geometry_msgs.msg import Twist
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Bool, String

from .emotion_logic import normalize_emotion


class EmotionBehaviorNode(Node):
    """Executes motion trajectories corresponding to recognized facial expressions."""

    def __init__(self):
        super().__init__('emotion_behavior_node')

        defaults = {
            'cmd_vel_topic': '/cmd_vel',
            'emotion_topic': '/face_emotion',
            'control_state_topic': '/emotion_control_state',
            'min_confidence': 0.60,
            'linear_speed': 0.15,
            'angular_speed': 0.40,
            'gentle_speed': 0.10,
            'rotation_speed': 0.60,
            'rotation_scale': 1.0,
            'max_linear_speed': 0.25,
            'max_angular_speed': 0.90,
            'linear_accel_limit': 0.35,
            'angular_accel_limit': 0.90,
            'action_timeout': 20.0,
            'watchdog_timeout': 1.0,
            'control_rate': 20.0,
            'continuous_mode': True,
            'curve_duration': 3.0,
            'gentle_duration': 2.0,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

        p = lambda name: self.get_parameter(name).value

        # Kinematic limits
        self.linear_speed = float(p('linear_speed'))
        self.angular_speed = float(p('angular_speed'))
        self.gentle_speed = float(p('gentle_speed'))
        self.rotation_speed = float(p('rotation_speed'))
        self.rotation_scale = float(p('rotation_scale'))

        self.max_linear_speed = float(p('max_linear_speed'))
        self.max_angular_speed = float(p('max_angular_speed'))
        self.linear_accel = float(p('linear_accel_limit'))
        self.angular_accel = float(p('angular_accel_limit'))

        self.min_confidence = float(p('min_confidence'))
        self.watchdog_timeout = float(p('watchdog_timeout'))
        self.action_timeout = float(p('action_timeout'))
        self.continuous_mode = bool(p('continuous_mode'))

        # Duration for discrete 360 degree spin maneuver (Surprise)
        # Angle = 2 * pi, time = (2 * pi / rotation_speed) * scale
        self.rotation_duration = (2.0 * math.pi / max(0.01, self.rotation_speed)) * self.rotation_scale

        self.durations = {
            'happy': float(p('curve_duration')),
            'sad': float(p('curve_duration')),
            'angry': float(p('gentle_duration')),
            'fear': float(p('gentle_duration')),
            'surprise': self.rotation_duration,
        }

        # Publishers & Subscribers
        self.pub = self.create_publisher(Twist, p('cmd_vel_topic'), 10)
        self.state_pub = self.create_publisher(String, p('control_state_topic'), 10)
        self.sub = self.create_subscription(String, p('emotion_topic'), self._emotion_callback, 10)
        self.estop_sub = self.create_subscription(Bool, '/emergency_stop', self._estop_callback, 10)

        # Internal State Machine
        self.last_message_time = time.monotonic()
        self.current_emotion = None
        self.current_confidence = 0.0
        self.state = 'WAITING_FOR_FACE'

        # Special maneuver latch (Surprise: 360-spin)
        self.active_maneuver = None
        self.maneuver_started = 0.0
        self.surprise_completed = False
        self.last_discrete_emotion = None

        self.emergency_stop = False
        self.current_linear = 0.0
        self.current_angular = 0.0

        # Control Loop
        hz = max(5.0, float(p('control_rate')))
        self.dt = 1.0 / hz
        self.timer = self.create_timer(self.dt, self._control_loop)

        self.get_logger().info(
            f'Emotion Behavior Node initialized on {p("cmd_vel_topic")} '
            f'(Continuous: {self.continuous_mode}, Spin duration: {self.rotation_duration:.2f}s)'
        )

    def _set_state(self, state):
        if self.state != state:
            self.state = state
            self.state_pub.publish(String(data=state))

    def _stop_now(self, state_name):
        self.active_maneuver = None
        self._set_state(state_name)
        self.current_linear = 0.0
        self.current_angular = 0.0
        self._publish_velocity(0.0, 0.0)

    def _estop_callback(self, msg):
        if msg.data:
            self.emergency_stop = True
            self.active_maneuver = None
            self._stop_now('EMERGENCY_STOP_LATCHED')
            self.get_logger().error('Emergency Stop triggered!')
        else:
            if self.emergency_stop:
                self.emergency_stop = False
                self.active_maneuver = None
                self.surprise_completed = False
                self._stop_now('WAITING_FOR_FACE')
                self.get_logger().info('Emergency Stop released.')

    def _emotion_callback(self, msg):
        self.last_message_time = time.monotonic()

        try:
            payload = json.loads(msg.data)
            emotion = normalize_emotion(payload.get('emotion'))
            confidence = float(payload.get('confidence', 0.0))
        except (ValueError, TypeError, AttributeError):
            emotion, confidence = None, 0.0

        if self.emergency_stop:
            return

        # Check face detection and confidence threshold
        if not emotion or confidence < self.min_confidence:
            self.current_emotion = None
            self.current_confidence = 0.0
            if self.active_maneuver is None:
                self._stop_now('NO_FACE' if not emotion else 'LOW_CONFIDENCE')
            return

        self.current_emotion = emotion
        self.current_confidence = confidence

        # Neutral expression -> Stop in place
        if emotion == 'neutral':
            self.active_maneuver = None
            self.surprise_completed = False
            self._stop_now('STOPPED_NEUTRAL')
            return

        # Surprise expression -> Execute 360-degree rotation maneuver
        if emotion == 'surprise':
            if not self.surprise_completed and self.active_maneuver != 'surprise':
                self.active_maneuver = 'surprise'
                self.maneuver_started = time.monotonic()
                self._set_state('SPINNING_360_SURPRISE')
                self.get_logger().info(
                    f'[SURPRISE] Starting 360-degree rotation maneuver '
                    f'({self.rotation_duration:.2f}s at {self.rotation_speed:.2f} rad/s)'
                )
            return
        else:
            # Leaving surprise resets the completed flag
            self.surprise_completed = False

        # If surprise rotation is in progress, let it complete
        if self.active_maneuver == 'surprise':
            return

        if self.continuous_mode:
            self._set_state(f'RUNNING_{emotion.upper()}')
        else:
            # Discrete / bounded duration mode
            if emotion != self.last_discrete_emotion or self.active_maneuver is None:
                self.last_discrete_emotion = emotion
                self.active_maneuver = emotion
                self.maneuver_started = time.monotonic()
                self._set_state(f'RUNNING_{emotion.upper()}')

    def _compute_target_velocity(self):
        """Map active emotion or maneuver to (linear_x, angular_z) command."""
        if self.emergency_stop:
            return 0.0, 0.0

        now = time.monotonic()

        # Watchdog: stop if no fresh perception message arrives
        if now - self.last_message_time > self.watchdog_timeout:
            self.active_maneuver = None
            return 0.0, 0.0

        # Discrete Surprise 360-degree maneuver
        if self.active_maneuver == 'surprise':
            elapsed = now - self.maneuver_started
            if elapsed >= self.rotation_duration:
                self.active_maneuver = None
                self.surprise_completed = True
                self._set_state('DONE_SURPRISE')
                self.get_logger().info('[SURPRISE] 360-degree rotation completed.')
                return 0.0, 0.0
            return 0.0, self.rotation_speed

        # Continuous mode
        if self.continuous_mode:
            if not self.current_emotion or self.current_emotion == 'neutral':
                return 0.0, 0.0

            # Target velocity table:
            # Happy: Curve left (v > 0, w > 0)
            # Sad: Curve right (v > 0, w < 0)
            # Angry: Move backward gently (v < 0, w = 0)
            # Fear: Move forward gently (v > 0, w = 0)
            mapping = {
                'happy': (self.linear_speed, self.angular_speed),
                'sad': (self.linear_speed, -self.angular_speed),
                'angry': (-self.gentle_speed, 0.0),
                'fear': (self.gentle_speed, 0.0),
            }
            return mapping.get(self.current_emotion, (0.0, 0.0))

        # Discrete bounded mode
        if self.active_maneuver:
            elapsed = now - self.maneuver_started
            duration = self.durations.get(self.active_maneuver, 2.0)
            if elapsed >= duration:
                done = self.active_maneuver
                self.active_maneuver = None
                self._set_state(f'DONE_{done.upper()}')
                return 0.0, 0.0

            mapping = {
                'happy': (self.linear_speed, self.angular_speed),
                'sad': (self.linear_speed, -self.angular_speed),
                'angry': (-self.gentle_speed, 0.0),
                'fear': (self.gentle_speed, 0.0),
            }
            return mapping.get(self.active_maneuver, (0.0, 0.0))

        return 0.0, 0.0

    @staticmethod
    def _approach(current, target, max_delta):
        """Ramp acceleration limiting to prevent abrupt jerk."""
        if current < target:
            return min(current + max_delta, target)
        return max(current - max_delta, target)

    def _publish_velocity(self, linear, angular):
        """Clamp velocity to hardware limits and publish geometry_msgs/Twist."""
        linear = max(-self.max_linear_speed, min(self.max_linear_speed, float(linear)))
        angular = max(-self.max_angular_speed, min(self.max_angular_speed, float(angular)))

        msg = Twist()
        msg.linear.x = linear
        msg.angular.z = angular
        self.pub.publish(msg)

    def _control_loop(self):
        now = time.monotonic()
        timed_out = (now - self.last_message_time) > self.watchdog_timeout

        if timed_out and not self.emergency_stop:
            self.active_maneuver = None
            self._set_state('WATCHDOG_STOP')

        target_linear, target_angular = self._compute_target_velocity()

        if self.emergency_stop or timed_out:
            self.current_linear = 0.0
            self.current_angular = 0.0
            self._publish_velocity(0.0, 0.0)
        else:
            # Apply smooth acceleration limits
            self.current_linear = self._approach(
                self.current_linear, target_linear, self.linear_accel * self.dt
            )
            self.current_angular = self._approach(
                self.current_angular, target_angular, self.angular_accel * self.dt
            )
            self._publish_velocity(self.current_linear, self.current_angular)

    def destroy_node(self):
        try:
            self.active_maneuver = None
            self._publish_velocity(0.0, 0.0)
        finally:
            super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = EmotionBehaviorNode()
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
