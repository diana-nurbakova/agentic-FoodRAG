"""Counterbalancing & between-subjects assignment.

- ``assign_profile``: round-robin across active profiles (balanced group sizes).
- ``get_plan_order``: balanced Latin square so each plan appears in each
  position equally often (clean position-effect analysis).

In production the participant number comes from a shared counter in the config
service (HF Dataset repo) so assignment is globally balanced. In this demo we
derive a stable per-session number from the session id (see ``state.py``), which
gives variety across sessions but not guaranteed global balance.
"""

from __future__ import annotations

# 3×3 balanced Latin square
LATIN_SQUARE_3 = [[0, 1, 2], [1, 2, 0], [2, 0, 1]]
# 4×4 balanced Latin square (Williams design)
LATIN_SQUARE_4 = [[0, 1, 3, 2], [1, 2, 0, 3], [2, 3, 1, 0], [3, 0, 2, 1]]


def get_plan_order(participant_number: int, num_plans: int = 3) -> list[int]:
    """Return the counterbalanced plan order for a participant."""
    square = LATIN_SQUARE_4 if num_plans == 4 else LATIN_SQUARE_3
    return square[participant_number % len(square)]


def assign_profile_index(participant_number: int, num_profiles: int) -> int:
    """Round-robin profile index (between-subjects)."""
    return participant_number % num_profiles


def latin_square_group(participant_number: int, num_plans: int = 3) -> int:
    """The Latin-square group (row) the participant belongs to."""
    square = LATIN_SQUARE_4 if num_plans == 4 else LATIN_SQUARE_3
    return participant_number % len(square)
