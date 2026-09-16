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

from .observer.ball_placement_observer import BallPlacementObserver
from .observer.customized_observer import CustomizedObserver
from .observer.goal_observer import GoalObserver
from .observer.robot_speed import RobotSpeedObserver
from .field_geometry import FieldGeometry
from .vision_world import VisionWorld

from typing import Optional

from typing_extensions import deprecated


class WorldObserver:
    """
    Observes the world and updates the state of the world.
    """
    # These defaults describe a Division A field. They are only the shape the
    # observers take before the first geometry packet arrives; `update()`
    # replaces them with the real dimensions as soon as vision reports them, so
    # a Division B match is not scored against a Division A goal.
    def __init__(self, field_length: float = 12.0, field_width: float = 9.0,
                 goal_width: float = 1.8, goal_depth: float = 0.18):
        self._field_length = field_length
        self._field_half_length = field_length / 2.0
        self._field_width = field_width
        self._goal_width = goal_width
        self._goal_half_width = goal_width / 2.0
        self._goal_depth = goal_depth
        self._field_geometry: Optional[FieldGeometry] = None

        self._vision_world = VisionWorld()

        self._ball_placement_observer = BallPlacementObserver()
        self._goal_observer = GoalObserver(
            self._field_half_length, self._goal_half_width, self._goal_depth)
        self._robot_speed_observer = RobotSpeedObserver()
        self._customized_observer = CustomizedObserver()

    def update(self, vision_world: VisionWorld) -> None:
        self._vision_world = vision_world

        geometry = self._vision_world.get_field_geometry()
        if geometry is not None and geometry != self._field_geometry:
            self._apply_field_geometry(geometry)

        ball = self._vision_world.get_ball()
        blue_robots = self._vision_world.get_blue_robots()
        yellow_robots = self._vision_world.get_yellow_robots()
        timestamp = self._vision_world.get_timestamp()

        self._ball_placement_observer.update(ball, blue_robots, yellow_robots)
        self._goal_observer.update(ball)
        self._robot_speed_observer.update(blue_robots, yellow_robots, timestamp)
        self._customized_observer.update(ball, blue_robots, yellow_robots)

    def reset(self) -> None:
        self._ball_placement_observer.reset()
        self._goal_observer.reset()
        self._robot_speed_observer.reset()
        self._customized_observer.reset_results()

    def ball_placement(self) -> BallPlacementObserver:
        return self._ball_placement_observer

    def goal(self) -> GoalObserver:
        return self._goal_observer

    def robot_speed(self) -> RobotSpeedObserver:
        return self._robot_speed_observer

    def customized(self) -> CustomizedObserver:
        return self._customized_observer

    @deprecated('ball_has_been_in_positive_goal() is moved to goal(). Use goal() instead.')
    def ball_has_been_in_positive_goal(self) -> bool:
        return self._goal_observer.ball_has_been_in_positive_goal()

    @deprecated('ball_has_been_in_negative_goal() is moved to goal(). Use goal() instead.')
    def ball_has_been_in_negative_goal(self) -> bool:
        return self._goal_observer.ball_has_been_in_negative_goal()

    def get_world(self) -> VisionWorld:
        return self._vision_world

    def get_field_geometry(self) -> Optional[FieldGeometry]:
        """Field dimensions as last reported by vision, or None if none yet."""
        return self._field_geometry

    def _apply_field_geometry(self, geometry: FieldGeometry) -> None:
        self._field_geometry = geometry
        self._field_length = geometry.field_length
        self._field_half_length = geometry.half_length
        self._field_width = geometry.field_width
        self._goal_width = geometry.goal_width
        self._goal_half_width = geometry.half_goal_width
        self._goal_depth = geometry.goal_depth

        # GoalObserver holds the dimensions it was built with, so rebuild it.
        # Its latched "a goal has been scored" flags are deliberately not
        # carried over: they were decided against the wrong goal.
        self._goal_observer = GoalObserver(
            self._field_half_length, self._goal_half_width, self._goal_depth)
