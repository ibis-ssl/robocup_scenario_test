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

from rcst.proto.ssl_vision_wrapper_pb2 import SSL_WrapperPacket
from rcst.vision_world import VisionWorld


def make_packet(t_capture, blue_ids=(), yellow_ids=(), x=0.0, y=0.0):
    """Build a single-camera detection frame containing the given robots."""
    packet = SSL_WrapperPacket()
    detection = packet.detection
    detection.frame_number = 0
    detection.t_capture = t_capture
    detection.t_sent = t_capture
    detection.camera_id = 0

    for robot_id in blue_ids:
        robot = detection.robots_blue.add()
        robot.confidence = 1.0
        robot.robot_id = robot_id
        robot.x = x
        robot.y = y
        robot.pixel_x = x
        robot.pixel_y = y

    for robot_id in yellow_ids:
        robot = detection.robots_yellow.add()
        robot.confidence = 1.0
        robot.robot_id = robot_id
        robot.x = x
        robot.y = y
        robot.pixel_x = x
        robot.pixel_y = y

    return packet.SerializeToString()


def test_robots_that_leave_the_field_are_removed():
    # A robot that stops being detected must not linger at its last position:
    # scenario tests compare every robot against every other one, so a leftover
    # entry reads as a real robot standing where nothing is.
    world = VisionWorld(robot_timeout=0.5)

    world.update_with_vision_packet(
        make_packet(0.0, blue_ids=range(11), yellow_ids=range(11)))
    assert sorted(world.get_blue_robots().keys()) == list(range(11))
    assert sorted(world.get_yellow_robots().keys()) == list(range(11))

    # Only robots 0-2 stay on the field, and enough time passes for the rest to
    # expire.
    for step in range(40):
        world.update_with_vision_packet(
            make_packet(0.016 * (step + 1), blue_ids=(0, 1, 2), yellow_ids=(0, 1, 2)))

    assert sorted(world.get_blue_robots().keys()) == [0, 1, 2]
    assert sorted(world.get_yellow_robots().keys()) == [0, 1, 2]


def test_robot_missing_for_less_than_the_timeout_is_kept():
    # One camera's frame carries only the robots that camera saw, and detections
    # are dropped now and then, so a short absence must not remove a robot.
    world = VisionWorld(robot_timeout=0.5)

    world.update_with_vision_packet(make_packet(0.0, blue_ids=(0, 1)))
    assert sorted(world.get_blue_robots().keys()) == [0, 1]

    # Robot 1 is absent for 0.4 s, which is within the timeout.
    for step in range(25):
        world.update_with_vision_packet(
            make_packet(0.016 * (step + 1), blue_ids=(0,)))

    assert sorted(world.get_blue_robots().keys()) == [0, 1]


def test_robot_keeps_its_last_position_while_it_is_missing():
    world = VisionWorld(robot_timeout=0.5)

    world.update_with_vision_packet(make_packet(0.0, blue_ids=(0,), x=1000.0, y=2000.0))
    world.update_with_vision_packet(make_packet(0.1, blue_ids=()))

    robot = world.get_blue_robots()[0]
    assert robot.x == 1.0
    assert robot.y == 2.0


def test_robot_that_comes_back_is_tracked_again():
    world = VisionWorld(robot_timeout=0.5)

    world.update_with_vision_packet(make_packet(0.0, blue_ids=(0, 1)))
    world.update_with_vision_packet(make_packet(1.0, blue_ids=(0,)))
    assert sorted(world.get_blue_robots().keys()) == [0]

    world.update_with_vision_packet(make_packet(1.1, blue_ids=(0, 1), x=3000.0))
    assert sorted(world.get_blue_robots().keys()) == [0, 1]
    assert world.get_blue_robots()[1].x == 3.0


def test_robots_placed_directly_are_not_removed():
    # test_world_observer.py populates the dicts by hand; those robots have no
    # detection time and must be left alone.
    from rcst.robot import Robot

    world = VisionWorld(robot_timeout=0.5)
    world._blue_robots[7] = Robot(x=0.0, y=0.0, id=7)

    world.update_with_vision_packet(make_packet(0.0, blue_ids=(0,)))
    world.update_with_vision_packet(make_packet(5.0, blue_ids=(0,)))

    assert sorted(world.get_blue_robots().keys()) == [0, 7]


def test_timestamp_comes_from_the_newest_camera():
    # Expiry is measured against the newest capture time from any camera; a
    # lagging camera must not expire robots the others still see.
    world = VisionWorld(robot_timeout=0.5)

    world.update_with_vision_packet(make_packet(1.0, blue_ids=(0,)))
    world.update_with_vision_packet(make_packet(0.9, blue_ids=(1,)))

    assert world.get_timestamp() == 1.0
    assert sorted(world.get_blue_robots().keys()) == [0, 1]
