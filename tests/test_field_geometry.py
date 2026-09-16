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

import pytest

from rcst.field_geometry import FieldGeometry
from rcst.proto.ssl_vision_wrapper_pb2 import SSL_WrapperPacket
from rcst.vision_world import VisionWorld
from rcst.world_observer import WorldObserver


# The dimensions the ER-Force simulator reports for `-g 2020B` (Division B) and
# `-g 2023` (Division A), in mm, as captured from a real vision log.
DIV_B = dict(field_length=9000, field_width=6000, goal_width=1000,
             goal_depth=180, boundary_width=300, penalty_x=3500, penalty_y=1000)
DIV_A = dict(field_length=12000, field_width=9000, goal_width=1800,
             goal_depth=180, boundary_width=250, penalty_x=4200, penalty_y=1800)


def make_geometry_packet(dims, with_penalty_lines=True, explicit_penalty=False):
    packet = SSL_WrapperPacket()
    field = packet.geometry.field
    field.field_length = dims["field_length"]
    field.field_width = dims["field_width"]
    field.goal_width = dims["goal_width"]
    field.goal_depth = dims["goal_depth"]
    field.boundary_width = dims["boundary_width"]

    if explicit_penalty:
        field.penalty_area_depth = \
            dims["field_length"] // 2 - dims["penalty_x"]
        field.penalty_area_width = dims["penalty_y"] * 2

    if with_penalty_lines:
        line = field.field_lines.add()
        line.name = "RightPenaltyStretch"
        line.p1.x = float(dims["penalty_x"])
        line.p1.y = float(-dims["penalty_y"])
        line.p2.x = float(dims["penalty_x"])
        line.p2.y = float(dims["penalty_y"])
        line.thickness = 10.0

    return packet.SerializeToString()


def test_division_b_geometry_is_parsed_in_metres():
    world = VisionWorld()
    world.update_with_vision_packet(make_geometry_packet(DIV_B))

    geometry = world.get_field_geometry()
    assert geometry.field_length == 9.0
    assert geometry.field_width == 6.0
    assert geometry.half_length == 4.5
    assert geometry.half_width == 3.0
    assert geometry.goal_width == 1.0
    assert geometry.boundary_width == 0.3
    # The wall, not the goal line, is what a teleport gets clamped against.
    assert geometry.wall_x == pytest.approx(4.8)
    assert geometry.wall_y == pytest.approx(3.3)


def test_division_a_and_b_differ_where_it_matters():
    # A coordinate that sits inside Division A can be outside Division B, which
    # is what made scenario tests written for the wrong division fail.
    world_a, world_b = VisionWorld(), VisionWorld()
    world_a.update_with_vision_packet(make_geometry_packet(DIV_A))
    world_b.update_with_vision_packet(make_geometry_packet(DIV_B))

    geometry_a = world_a.get_field_geometry()
    geometry_b = world_b.get_field_geometry()

    assert geometry_a.wall_x == pytest.approx(6.25)
    assert geometry_b.wall_x == pytest.approx(4.8)
    assert geometry_a != geometry_b


def test_penalty_area_comes_from_the_boundary_lines():
    world = VisionWorld()
    world.update_with_vision_packet(make_geometry_packet(DIV_B))

    geometry = world.get_field_geometry()
    assert geometry.penalty_area_x() == pytest.approx(3.5)
    assert geometry.penalty_half_width_or_raise() == pytest.approx(1.0)
    assert geometry.is_in_penalty_area(4.0, 0.5)
    assert geometry.is_in_penalty_area(-4.0, -0.5)
    assert not geometry.is_in_penalty_area(3.0, 0.5)
    assert not geometry.is_in_penalty_area(4.0, 1.5)


def test_explicit_penalty_fields_are_preferred_when_present():
    world = VisionWorld()
    world.update_with_vision_packet(
        make_geometry_packet(DIV_B, with_penalty_lines=False,
                             explicit_penalty=True))

    geometry = world.get_field_geometry()
    assert geometry.penalty_area_x() == pytest.approx(3.5)
    assert geometry.penalty_half_width_or_raise() == pytest.approx(1.0)


def test_missing_penalty_area_raises_instead_of_guessing():
    # Guessing a penalty area is how a test ends up asserting against a box
    # that is not on the field it is running on.
    world = VisionWorld()
    world.update_with_vision_packet(
        make_geometry_packet(DIV_B, with_penalty_lines=False))

    geometry = world.get_field_geometry()
    assert geometry.penalty_depth is None
    with pytest.raises(ValueError):
        geometry.penalty_area_x()
    with pytest.raises(ValueError):
        geometry.is_in_penalty_area(4.0, 0.0)


def test_geometry_is_none_before_any_packet_arrives():
    assert VisionWorld().get_field_geometry() is None
    assert WorldObserver().get_field_geometry() is None


def test_observer_rebuilds_the_goal_from_the_reported_geometry():
    # WorldObserver defaults to Division A; a Division B match must not be
    # scored against a Division A goal.
    observer = WorldObserver()
    assert observer._field_half_length == 6.0
    assert observer._goal_half_width == 0.9

    world = VisionWorld()
    world.update_with_vision_packet(make_geometry_packet(DIV_B))
    observer.update(world)

    assert observer.get_field_geometry() == FieldGeometry(
        field_length=9.0, field_width=6.0, goal_width=1.0, goal_depth=0.18,
        boundary_width=0.3, penalty_depth=1.0, penalty_half_width=1.0)
    assert observer._field_half_length == 4.5
    assert observer._goal_half_width == 0.5
    assert observer.goal()._field_half_length == 4.5
    assert observer.goal()._goal_half_width == 0.5


def test_a_detection_only_packet_leaves_the_geometry_alone():
    world = VisionWorld()
    world.update_with_vision_packet(make_geometry_packet(DIV_B))

    packet = SSL_WrapperPacket()
    packet.detection.frame_number = 0
    packet.detection.t_capture = 1.0
    packet.detection.t_sent = 1.0
    packet.detection.camera_id = 0
    world.update_with_vision_packet(packet.SerializeToString())

    assert world.get_field_geometry().field_length == 9.0
