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

import dataclasses
from typing import Optional

from .proto.ssl_vision_geometry_pb2 import SSL_GeometryFieldSize


# Names of the penalty area boundary lines in SSL_GeometryFieldSize.field_lines.
#
# The line runs along the front edge of one penalty area, so a single line gives
# both the depth (how far it stands from the goal line) and the half width (how
# far it extends either side of the centre line).
_PENALTY_STRETCH_NAMES = ("RightPenaltyStretch", "LeftPenaltyStretch")


@dataclasses.dataclass(frozen=True)
class FieldGeometry:
    """Field dimensions taken from the vision geometry packet, in metres.

    Scenario tests must not hard-code a division's dimensions: the same test is
    run against Division A (12.0 x 9.0) and Division B (9.0 x 6.0) fields, and a
    coordinate that is comfortably inside one field is outside the other. The
    simulator silently clamps a teleport that lands outside the field, so a test
    written for the wrong division does not fail with an error -- it runs against
    a world that is not the one it asked for.

    `penalty_depth` and `penalty_half_width` are None when the geometry packet
    carries no penalty area lines; ask for them through `penalty_area_x()` and
    `penalty_half_width_or_raise()`, which say so rather than guessing.
    """

    field_length: float
    field_width: float
    goal_width: float
    goal_depth: float
    boundary_width: float
    penalty_depth: Optional[float] = None
    penalty_half_width: Optional[float] = None

    @classmethod
    def from_proto(cls, field: SSL_GeometryFieldSize) -> 'FieldGeometry':
        penalty_depth, penalty_half_width = _parse_penalty_area(field)
        return cls(
            field_length=field.field_length * 0.001,
            field_width=field.field_width * 0.001,
            goal_width=field.goal_width * 0.001,
            goal_depth=field.goal_depth * 0.001,
            boundary_width=field.boundary_width * 0.001,
            penalty_depth=penalty_depth,
            penalty_half_width=penalty_half_width,
        )

    @property
    def half_length(self) -> float:
        """Distance from the centre to a goal line."""
        return self.field_length / 2.0

    @property
    def half_width(self) -> float:
        """Distance from the centre to a touch line."""
        return self.field_width / 2.0

    @property
    def half_goal_width(self) -> float:
        return self.goal_width / 2.0

    @property
    def wall_x(self) -> float:
        """Distance from the centre to the wall behind a goal line."""
        return self.half_length + self.boundary_width

    @property
    def wall_y(self) -> float:
        """Distance from the centre to the wall behind a touch line."""
        return self.half_width + self.boundary_width

    def penalty_area_x(self) -> float:
        """Absolute x of the penalty area's front edge."""
        if self.penalty_depth is None:
            raise ValueError(
                "The geometry packet carries no penalty area lines, so the "
                "penalty area cannot be derived. Expected one of {} in "
                "field_lines.".format(", ".join(_PENALTY_STRETCH_NAMES)))
        return self.half_length - self.penalty_depth

    def penalty_half_width_or_raise(self) -> float:
        if self.penalty_half_width is None:
            raise ValueError(
                "The geometry packet carries no penalty area lines, so the "
                "penalty area cannot be derived. Expected one of {} in "
                "field_lines.".format(", ".join(_PENALTY_STRETCH_NAMES)))
        return self.penalty_half_width

    def is_in_penalty_area(self, x: float, y: float) -> bool:
        """Whether (x, y) is inside either penalty area."""
        return abs(x) >= self.penalty_area_x() \
            and abs(y) <= self.penalty_half_width_or_raise()


def _parse_penalty_area(field: SSL_GeometryFieldSize):
    # The explicit fields are optional and some sources (the ER-Force simulator
    # among them) leave them unset, so fall back to the boundary lines, which
    # every source draws.
    if field.HasField("penalty_area_depth") and field.HasField("penalty_area_width"):
        # penalty_area_width is the full width, measured across the centre line.
        return (field.penalty_area_depth * 0.001,
                field.penalty_area_width * 0.0005)

    for line in field.field_lines:
        if line.name not in _PENALTY_STRETCH_NAMES:
            continue
        # The stretch is the line parallel to the goal line, so both endpoints
        # share an x and straddle y = 0.
        depth = field.field_length / 2.0 - abs(line.p1.x)
        half_width = max(abs(line.p1.y), abs(line.p2.y))
        return depth * 0.001, half_width * 0.001
    return None, None
