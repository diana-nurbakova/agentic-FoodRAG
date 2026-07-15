"""Build study meal-plan artifacts by SELECTING from already-run evaluation results.

The recommendations were computed offline (full-day evaluation runs); this script
does **not** re-run the pipeline. For each study profile (a chosen evaluation
question) and each condition (an eval result file), it reads the
``{meal: food.com URL}`` plan, resolves every URL to full recipe detail from HUMMUS
(ingredients, directions, per-serving nutrition), and writes one JSON artifact per
profile to ``ui/study_plans/{id}.json``. The study flow
(``ui.study_data.load_study_plans``) loads these.

Usage::

    uv run python scripts/build_study_plans.py --reselect   # select + build in one pass (full regen)
    uv run python scripts/build_study_plans.py              # build from the configured profiles
    uv run python scripts/build_study_plans.py --select     # only (re)select profiles.json, no build

Selection is diverse, not-"too-easy", **and** calorie-plausible (questions whose plans
contain corrupt-serving-size recipes in any condition are excluded).

The condition → result-file mapping is ``ui.study_data.get_study_conditions()``; the
profile → question mapping is ``ui.study_data.get_study_profiles()`` (both admin-editable
and persisted). ``--select`` seeds the profile default from a diverse, not-"too-easy"
set of questions.

Requires ``data/hummus`` present. No LLM key needed — this is a pure lookup/join.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "src"):
    _sp = str(_p)
    if _sp not in sys.path:
        sys.path.insert(0, _sp)

from ui.services import mapping, study_results  # noqa: E402
from ui.study_data import (  # noqa: E402
    STUDY_PLANS_DIR,
    get_study_conditions,
    get_study_profiles,
)

PROFILES_FILE = STUDY_PLANS_DIR / "profiles.json"


# ── HUMMUS recipe lookup (processed shape recipe_to_card expects, keeping url) ──
def load_recipes_by_url():
    """HUMMUS recipes indexed by food.com URL, in the processed column shape the GUI
    mapping expects (``calories [kcal]``, ``ingredient_quantities``, ``duration [min]``
    …). Mirrors ``ui.services.backend._build_engine`` but keeps ``recipe_url``."""
    from foodrag.config import settings as foodrag_settings
    from foodrag.food_rag.utils import hummus_docify
    from foodrag.utils import hummus_util, load_util

    data_path = str(foodrag_settings.repo_root() / "data")
    print("Loading HUMMUS recipes (one-time, ~1-2 min) …", flush=True)
    hummus_data = load_util.get_current_hummus(data_path)
    df = hummus_data["recipes_df"].copy()
    df = hummus_docify.stringify_nested_lists(df)
    df = hummus_docify.stringify_nested_dicts(df)
    df = df.rename(columns={
        "duration": "duration [min]", "ingredients": "ingredient_quantities",
        "ingredient_names": "ingredients", "direction_size": "number_of_directions",
        "ingredients_sizes": "number_of_ingredients",
    })
    df = hummus_util.clean_serves_to_int(df)
    df = df[df["recipe_url"].notna()].drop_duplicates("recipe_url")
    print(f"  {len(df):,} recipes indexed by URL.", flush=True)
    return df.set_index("recipe_url")


_GUI_MEALS = [("breakfast", "Breakfast"), ("lunch", "Lunch"), ("dinner", "Dinner"), ("snack", "Snack")]

# Plausibility bounds for per-serving calories. HUMMUS has recipes with corrupt
# serving sizes (e.g. a 2,559 g "Green Pepper Jelly" → 82,712 kcal/serving); such
# recipes make broken study stimuli, so questions whose plans contain them (in any
# condition) are excluded during selection.
MEAL_KCAL_MAX = 2000        # a single realistic serving never exceeds this
DAY_KCAL_MIN, DAY_KCAL_MAX = 600, 5500


def _plan_kcal(plan_urls: dict, recipes_by_url) -> tuple[int, int]:
    """(daily total, largest single meal) per-serving kcal for a ``{meal: url}`` plan."""
    cals = []
    for meal_key, meal_label in _GUI_MEALS:
        url = plan_urls.get(meal_key)
        if url and url in recipes_by_url.index:
            recipe = recipes_by_url.loc[url].to_dict()
            recipe["recipe_url"] = url
            cals.append(int(mapping.recipe_to_card(recipe, meal_label).get("cal") or 0))
    return (sum(cals), max(cals) if cals else 0)


def _make_plan_ok(recipes_by_url):
    """Predicate: a plan's per-serving calories are physiologically plausible."""
    def plan_ok(plan_urls: dict) -> bool:
        day, meal_max = _plan_kcal(plan_urls, recipes_by_url)
        return DAY_KCAL_MIN <= day <= DAY_KCAL_MAX and meal_max <= MEAL_KCAL_MAX
    return plan_ok


def select_profiles(n: int, min_distinct: int, recipes_by_url) -> list[dict]:
    """Diverse, not-too-easy, calorie-plausible profiles across the current conditions."""
    files = [c["config"]["result_file"] for c in get_study_conditions()]
    return study_results.select_default_profiles(
        files, n=n, min_distinct=min_distinct, plan_ok=_make_plan_ok(recipes_by_url))


def _cards_from_plan(plan: dict, recipes_by_url) -> tuple[list[dict], list[str]]:
    """Resolve a ``{meal: url}`` plan to GUI recipe cards; report unresolved URLs."""
    cards, missing = [], []
    for meal_key, meal_label in _GUI_MEALS:
        url = plan.get(meal_key)
        if not url:
            continue
        if url not in recipes_by_url.index:
            missing.append(url)
            continue
        recipe = recipes_by_url.loc[url].to_dict()
        recipe["recipe_url"] = url  # ensure the food.com link survives into the card
        card = {"meal": meal_label, **mapping.recipe_to_card(recipe, meal_label)}
        cards.append(card)
    return cards, missing


def _plan_record(cond: dict, profile: dict, recipes_by_url) -> dict:
    """Build one unified study-plan dict for a (condition, profile) pair."""
    config = cond["config"]
    result_file = config["result_file"]
    qid = profile["question_id"]
    plan_urls = study_results.plan_for(result_file, qid)
    cards, missing = _cards_from_plan(plan_urls, recipes_by_url)
    explanation = study_results.explanation_for(result_file, qid)
    if missing:
        print(f"    ! {cond['cond']}: {len(missing)} recipe URL(s) not found in HUMMUS", flush=True)
    return {
        "cond": cond["cond"],
        "label": cond["label"],
        "config": config,                       # hidden in GUI, logged for analysis
        "question_id": qid,
        "explanation": {"summary": explanation or "No explanation was recorded for this plan.",
                        "reasons": []},
        "recipes": cards,
        "total_kcal": sum(int(c.get("cal") or 0) for c in cards),
        "source": "eval_results",
    }


def build_profile(profile: dict, conditions: list[dict], recipes_by_url) -> dict:
    print(f"Profile {profile['id']} — {profile.get('goal', '')} (question {profile['question_id']})",
          flush=True)
    plans = []
    for cond in conditions:
        rec = _plan_record(cond, profile, recipes_by_url)
        meal_max = max((int(r.get("cal") or 0) for r in rec["recipes"]), default=0)
        warn = ("  ⚠ implausible calories (corrupt serving size?)"
                if not (DAY_KCAL_MIN <= rec["total_kcal"] <= DAY_KCAL_MAX) or meal_max > MEAL_KCAL_MAX
                else "")
        print(f"  [{cond['cond']}] {cond['config']['config_id']}: "
              f"{len(rec['recipes'])} meals, {rec['total_kcal']} kcal{warn}", flush=True)
        plans.append(rec)
    return {"profile_id": profile["id"], "question_id": profile["question_id"],
            "profile_prompt": profile.get("prompt", ""), "plans": plans}


# ── selection step ─────────────────────────────────────────────────────────
def _report_degenerate(files: list[str]) -> None:
    """Print per-condition plan health to explain a 0-candidate selection."""
    meals = ["breakfast", "lunch", "dinner", "snack"]
    print("Per-condition plan health (why no candidates?):", flush=True)
    for f in files:
        try:
            df = study_results.load_results(f)
        except Exception as exc:  # noqa: BLE001
            print(f"  {f}: unreadable ({exc})", flush=True)
            continue
        complete = distinct = 0
        for qid in df.index:
            p = study_results.plan_for(f, qid)
            if study_results.plan_is_complete(p):
                complete += 1
                if len({p.get(m) for m in meals if p.get(m)}) >= 3:
                    distinct += 1
        print(f"  {f}: {complete} complete, {distinct} non-degenerate "
              f"({'OK' if distinct else 'DEGENERATE — every plan repeats one recipe'})", flush=True)


def _write_profiles(profiles: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"profiles": profiles}, indent=2, ensure_ascii=False), encoding="utf-8")
    for p in profiles:
        print(f"  {p['id']}  {p['question_id']}  [{p['constraint_count']} constraints, "
              f"{p['answer_set_size']} valid recipes]  {p['goal']}", flush=True)
    print(f"Wrote {out}", flush=True)


def do_select(n: int, out: Path, min_distinct: int = 3) -> int:
    files = [c["config"]["result_file"] for c in get_study_conditions()]
    print(f"Selecting {n} diverse, plausible profiles from {len(files)} conditions …", flush=True)
    recipes_by_url = load_recipes_by_url()  # needed for the calorie-plausibility filter
    profiles = select_profiles(n, min_distinct, recipes_by_url)
    if not profiles:
        print("No candidate questions satisfy all conditions.", file=sys.stderr)
        _report_degenerate(files)
        print("Fix/replace any DEGENERATE condition file, then re-run.", file=sys.stderr)
        return 1
    _write_profiles(profiles, out)
    return 0


# ── build step ─────────────────────────────────────────────────────────────
def do_build(out_dir: Path, *, reselect: bool = False, n: int = 4, min_distinct: int = 3) -> int:
    conditions = get_study_conditions()
    recipes_by_url = load_recipes_by_url()
    if reselect:
        print(f"Re-selecting {n} diverse, plausible profiles …", flush=True)
        profiles = select_profiles(n, min_distinct, recipes_by_url)
        if not profiles:
            print("No candidate questions satisfy all conditions.", file=sys.stderr)
            _report_degenerate([c["config"]["result_file"] for c in conditions])
            return 1
        _write_profiles(profiles, out_dir / "profiles.json")
    else:
        profiles = get_study_profiles()
        if not profiles:
            print("No study profiles configured. Run with --reselect first.", file=sys.stderr)
            return 1
    out_dir.mkdir(parents=True, exist_ok=True)
    failures = 0
    for profile in profiles:
        try:
            data = build_profile(profile, conditions, recipes_by_url)
        except Exception as exc:  # noqa: BLE001 — keep going so one bad profile isn't fatal
            failures += 1
            print(f"  ! FAILED {profile.get('id')}: {type(exc).__name__}: {exc}", flush=True)
            continue
        out_path = out_dir / f"{profile['id']}.json"
        out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  wrote {out_path}", flush=True)
    if failures:
        print(f"Completed with {failures} failed profile(s).", flush=True)
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build study plans from evaluation results.")
    ap.add_argument("--select", action="store_true",
                    help="Only (re)select profiles and write profiles.json; do not build plans.")
    ap.add_argument("--reselect", action="store_true",
                    help="Reselect profiles AND build plans in one pass (recommended full regen).")
    ap.add_argument("--n", type=int, default=4, help="Number of profiles to select.")
    ap.add_argument("--min-distinct", type=int, default=3,
                    help="Min distinct recipes required per plan when selecting (default 3).")
    ap.add_argument("--out", type=Path, default=STUDY_PLANS_DIR, help="Output directory.")
    args = ap.parse_args(argv)

    if args.select:
        return do_select(args.n, args.out / "profiles.json", min_distinct=args.min_distinct)
    return do_build(args.out, reselect=args.reselect, n=args.n, min_distinct=args.min_distinct)


if __name__ == "__main__":
    raise SystemExit(main())
