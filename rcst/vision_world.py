# Copyright 2024 Roots
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List

from .ball import Ball
from .robot import Robot
from .robot import RobotDict

from .proto.ssl_vision_detection_pb2 import SSL_DetectionBall
from .proto.ssl_vision_detection_pb2 import SSL_DetectionFrame
from .proto.ssl_vision_detection_pb2 import SSL_DetectionRobot
from .proto.ssl_vision_wrapper_pb2 import SSL_WrapperPacket


# Seconds a robot is kept after its last detection.
#
# A detection frame only carries the robots one camera saw, so "absent from this
# frame" does not mean "gone from the field": with several cameras, or when
# detections are dropped, a robot is legitimately missing from many frames in a
# row. Robots are therefore removed on a timeout rather than per frame.
#
# SSL-Vision publishes at 60-75 Hz per camera, so this is roughly 30 frames of
# margin, while still dropping a robot that really left the field well within
# the sleep granularity a scenario test uses.
DEFAULT_ROBOT_TIMEOUT = 0.5


class VisionWorld:
    def __init__(self, robot_timeout: float = DEFAULT_ROBOT_TIMEOUT):
        self._ball: List[Ball] = [Ball()]
        self._blue_robots: RobotDict = {}
        self._yellow_robots: RobotDict = {}
        self._timestamp: float = 0.0
        self._robot_timeout = robot_timeout
        self._blue_last_seen: dict[int, float] = {}
        self._yellow_last_seen: dict[int, float] = {}

    def update_with_vision_packet(self, data: bytes) -> None:
        packet = SSL_WrapperPacket()
        try:
            packet.ParseFromString(data)
        except Exception as e:
            print("Failed to parse vision packet. Details: {}".format(e))
            return

        if packet.HasField("detection"):
            self._update_with_detection_frame(packet.detection)

    def get_ball(self) -> Ball:
        return self._ball[0]

    def get_blue_robots(self) -> RobotDict:
        return self._blue_robots

    def get_yellow_robots(self) -> RobotDict:
        return self._yellow_robots

    def get_timestamp(self) -> float:
        return self._timestamp

    def _update_with_detection_frame(self, detection: SSL_DetectionFrame) -> None:
        for ball in detection.balls:
            self._update_ball(ball)

        self._update_blue_robots(detection.robots_blue, detection.t_capture)
        self._update_yellow_robots(detection.robots_yellow, detection.t_capture)

        if detection.t_capture > self._timestamp:
            self._timestamp = detection.t_capture

        # Expiry is measured against the newest capture time seen from any
        # camera, not against this frame's own t_capture, so that a camera whose
        # clock lags slightly cannot expire robots the other cameras still see.
        self._remove_stale_robots(self._timestamp)

    def _update_ball(self, ball: SSL_DetectionBall) -> None:
        # Convert mm to m
        self._ball = [Ball(x=ball.x * 0.001,
                           y=ball.y * 0.001,)]

    def _update_blue_robots(
            self, robots: List[SSL_DetectionRobot], t_capture: float) -> None:
        for robot in robots:
            self._blue_robots[robot.robot_id] = Robot(x=robot.x * 0.001,
                                                      y=robot.y * 0.001,
                                                      orientation=robot.orientation,
                                                      id=robot.robot_id,
                                                      is_yellow=False)
            self._blue_last_seen[robot.robot_id] = t_capture

    def _update_yellow_robots(
            self, robots: List[SSL_DetectionRobot], t_capture: float) -> None:
        for robot in robots:
            self._yellow_robots[robot.robot_id] = Robot(x=robot.x * 0.001,
                                                        y=robot.y * 0.001,
                                                        orientation=robot.orientation,
                                                        id=robot.robot_id,
                                                        is_yellow=True)
            self._yellow_last_seen[robot.robot_id] = t_capture

    def _remove_stale_robots(self, now: float) -> None:
        self._remove_stale(self._blue_robots, self._blue_last_seen, now)
        self._remove_stale(self._yellow_robots, self._yellow_last_seen, now)

    def _remove_stale(
            self, robots: RobotDict, last_seen: dict[int, float], now: float) -> None:
        # Only robots this instance has actually seen in a detection frame are
        # candidates: robots placed directly into the dict (as some tests do)
        # have no last-seen time and are left untouched.
        expired = [robot_id for robot_id, seen in last_seen.items()
                   if now - seen > self._robot_timeout]
        for robot_id in expired:
            robots.pop(robot_id, None)
            last_seen.pop(robot_id, None)
