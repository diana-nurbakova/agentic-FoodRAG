"""Nutrition visual primitives: Nutri-Score strip, macro donut, nutrient
bars, triple health score and cooking-skill badge.

All functions return HTML strings (no Streamlit calls) so they can be embedded
inside larger component markup. Ported from the v4 mockup's NS / Donut / Bars /
TripleScore / Skill components. Colour AND letter are used for Nutri-Score so it
stays accessible without relying on colour alone (WCAG 1.4.1).
"""

from __future__ import annotations

import math

from ..theme import C, F, MACRO, NUTRI_C


def nutriscore_html(score: str, size: str = "normal") -> str:
    """Render the A–E Nutri-Score strip with the active grade enlarged."""
    s = (
        {"h": 14, "a": 18, "fs": 7, "af": 10, "g": 1}
        if size == "small"
        else {"h": 18, "a": 26, "fs": 8, "af": 13, "g": 2}
    )
    cells = ""
    for letter in ["A", "B", "C", "D", "E"]:
        active = letter == score
        dim = s["a"] if active else s["h"]
        cells += (
            f"<div style=\"width:{dim}px;height:{dim}px;border-radius:3px;"
            f"background:{NUTRI_C[letter]};opacity:{1 if active else 0.3};display:flex;"
            f"align-items:center;justify-content:center;color:#fff;"
            f"font-size:{s['af'] if active else s['fs']}px;font-weight:800;font-family:{F}\">{letter}</div>"
        )
    label = (
        f"<span style=\"font-family:{F};font-size:7px;font-weight:700;color:{C['textMuted']};"
        f"letter-spacing:1px;text-transform:uppercase;margin-bottom:1px\">Nutri-Score</span>"
        if size != "small"
        else ""
    )
    return (
        f"<div style=\"display:inline-flex;flex-direction:column\">{label}"
        f"<div style=\"display:flex;gap:{s['g']}px;align-items:center\">{cells}</div></div>"
    )


def skill_html(level: str) -> str:
    """Three-dot cooking-skill badge (Easy / Medium / Hard)."""
    dots = {"Easy": 1, "Medium": 2, "Hard": 3}.get(level, 1)
    col = {"Easy": C["nutA"], "Medium": C["nutC"], "Hard": C["nutE"]}.get(level, C["nutA"])
    pips = "".join(
        f"<div style=\"width:5px;height:5px;border-radius:50%;"
        f"background:{col if i <= dots else C['border']}\"></div>"
        for i in (1, 2, 3)
    )
    return (
        f"<div style=\"display:flex;align-items:center;gap:3px\">"
        f"<div style=\"display:flex;gap:2px\">{pips}</div>"
        f"<span style=\"font-size:9px;color:{C['textMuted']};font-family:{F}\">{level}</span></div>"
    )


def donut_html(protein: float, carbs: float, fat: float, cal: int, size: int = 48) -> str:
    """SVG macro donut with kcal in the centre."""
    total = protein + carbs + fat or 1
    p_p, p_c = protein / total, carbs / total
    r = size / 2 - 4
    cx = cy = size / 2

    def arc(start: float, pct: float, col: str) -> str:
        if pct <= 0:
            return ""
        a1 = start * 2 * math.pi - math.pi / 2
        a2 = (start + pct) * 2 * math.pi - math.pi / 2
        large = 1 if pct > 0.5 else 0
        x1, y1 = cx + r * math.cos(a1), cy + r * math.sin(a1)
        x2, y2 = cx + r * math.cos(a2), cy + r * math.sin(a2)
        return f'<path d="M {x1:.2f} {y1:.2f} A {r} {r} 0 {large} 1 {x2:.2f} {y2:.2f}" stroke="{col}" stroke-width="6" fill="none" />'

    arcs = (
        arc(0, p_p, MACRO["protein"])
        + arc(p_p, p_c, MACRO["carbs"])
        + arc(p_p + p_c, 1 - p_p - p_c, MACRO["fat"])
    )
    return (
        f"<div style=\"position:relative;width:{size}px;height:{size}px;flex-shrink:0\">"
        f"<svg width=\"{size}\" height=\"{size}\">{arcs}</svg>"
        f"<div style=\"position:absolute;inset:0;display:flex;align-items:center;justify-content:center;flex-direction:column\">"
        f"<span style=\"font-size:10px;font-weight:700;color:{C['text']};font-family:{F};line-height:1\">{cal}</span>"
        f"<span style=\"font-size:6px;color:{C['textMuted']};font-family:{F}\">kcal</span></div></div>"
    )


def bars_html(r: dict, expanded: bool = False) -> str:
    """Horizontal nutrient bars. Shows 3 macros collapsed, 6 when expanded."""
    items = [
        {"l": "Protein", "v": r["protein"], "m": 60, "u": "g", "c": MACRO["protein"]},
        {"l": "Carbs", "v": r["carbs"], "m": 80, "u": "g", "c": MACRO["carbs"]},
        {"l": "Fat", "v": r["fat"], "m": 40, "u": "g", "c": MACRO["fat"]},
        {"l": "Fiber", "v": r["fiber"], "m": 15, "u": "g", "c": C["nutA"]},
        {"l": "Sugar", "v": r["sugar"], "m": 30, "u": "g", "c": C["nutD"]},
        {"l": "Sodium", "v": r["sodium"], "m": 2400, "u": "mg", "c": C["nutE"]},
    ]
    shown = items if expanded else items[:3]
    rows = ""
    for it in shown:
        pct = min(it["v"] / it["m"] * 100, 100)
        rows += (
            f"<div style=\"display:flex;align-items:center;gap:4px\">"
            f"<span style=\"font-size:8px;color:{C['textMuted']};width:38px;font-family:{F}\">{it['l']}</span>"
            f"<div style=\"flex:1;height:4px;background:{C['tagBg']};border-radius:2px\">"
            f"<div style=\"height:4px;width:{pct:.0f}%;background:{it['c']};border-radius:2px\"></div></div>"
            f"<span style=\"font-size:8px;color:{C['textMuted']};width:32px;text-align:right;font-family:{F}\">{it['v']}{it['u']}</span></div>"
        )
    return f"<div style=\"display:flex;flex-direction:column;gap:3px\">{rows}</div>"


def triple_score_html(who: int, fsa: int, nutriscore: str) -> str:
    """Nutri-Score strip + WHO/10 + FSA numeric, colour-coded by healthiness."""
    who_c = C["nutA"] if who >= 7 else C["nutC"] if who >= 4 else C["nutE"]
    fsa_c = C["nutA"] if fsa <= 3 else C["nutC"] if fsa <= 8 else C["nutE"]
    return (
        f"<div style=\"display:flex;gap:8px;align-items:center;margin-top:4px\">"
        f"{nutriscore_html(nutriscore, 'small')}"
        f"<div style=\"display:flex;gap:6px;font-family:{F};font-size:9px;color:{C['textMuted']}\">"
        f"<span>WHO: <b style=\"color:{who_c}\">{who}/10</b></span>"
        f"<span>FSA: <b style=\"color:{fsa_c}\">{fsa}</b></span></div></div>"
    )
