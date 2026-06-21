"""Attention check item (instructed response, spec §5.4 / §5.8).

Renders like a normal Likert item but with an instructed-response prompt
("Please select 'Disagree'…"). Position is rotated per plan by the caller
(``study_data.plan_eval_items_for``). Returns (value, passed).
"""

from __future__ import annotations

from ..study_data import ATTENTION_ITEM
from ..theme import C, F, html
from .likert_scale import render_likert


def render_attention_check(key: str) -> tuple[int | None, bool]:
    """Render the attention check. Returns (selected value, passed)."""
    html(
        f"<div style=\"font-family:{F};font-size:10px;color:{C['warn']};font-weight:600;"
        f"text-transform:uppercase;letter-spacing:0.5px;margin-top:10px\">Attention check</div>"
    )
    value = render_likert(ATTENTION_ITEM["text"], key=key)
    passed = value == ATTENTION_ITEM["expected"]
    return value, passed
