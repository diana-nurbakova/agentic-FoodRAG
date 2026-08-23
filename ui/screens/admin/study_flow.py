"""Admin: Flow B study design — conditions & profiles.

The study's meal plans are **selected from already-run evaluation results** (no live
pipeline). Here the admin sets:

  - **Conditions** (C1/C2/C3): each maps to an eval result file (config). The config
    is recorded with every rating but never shown to participants.
  - **Profiles**: each maps to an evaluation ``question_id``. Defaults are chosen to be
    diverse and not "too easy"; the admin can swap question ids or re-suggest.

Changing either requires re-running ``scripts/build_study_plans.py`` to regenerate the
plan artifacts (this screen shows their status but does not run the heavy build).
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from ...admin_config import default_admin_config, get_admin_config
from ...services import config_service, study_results
from ...study_data import STUDY_PLANS_DIR, get_study_profiles


def _persistence_controls(cfg: dict) -> None:
    status = config_service.status()
    c1, c2, c3 = st.columns([1, 1, 2])
    if c1.button("💾 Save configuration", type="primary", key="sf_save"):
        ok, msg = config_service.save_config(cfg)
        (st.success if ok else st.error)(msg)
    if c2.button("↻ Reload from store", key="sf_reload"):
        persisted = config_service.load_persisted()
        if persisted is None:
            st.info("No persisted config found; keeping current settings.")
        else:
            st.session_state["admin_config"] = config_service.deep_merge(
                default_admin_config(), persisted)
            st.success("Reloaded persisted configuration.")
            st.rerun()
    dest = ("HF repo + local" if status["remote_enabled"] else "local file only")
    c3.caption(f"Persists to {dest}. After changes, re-run the build script to regenerate plans.")


def _render_conditions(cfg: dict) -> None:
    st.subheader("Conditions (hidden system configs)")
    st.caption("Each participant rates one plan per condition. Every condition maps to an "
               "already-run full-day result file — the recommendations are not regenerated live.")
    catalog = study_results.available_fullday_configs()
    by_id = {c["config_id"]: c for c in catalog}
    ids = sorted(by_id)
    if not catalog:
        st.warning("No full-day result files found under `eval_results/`.")
        return

    conditions = cfg.setdefault("study_conditions", [])
    for i, cond in enumerate(conditions):
        conf = cond.setdefault("config", {})
        cur = conf.get("config_id")
        col_a, col_b = st.columns([1, 3])
        with col_a:
            cond["label"] = st.text_input(f"{cond['cond']} label", value=cond.get("label", cond["cond"]),
                                          key=f"sf_lbl_{i}")
        with col_b:
            idx = ids.index(cur) if cur in ids else 0
            chosen = st.selectbox(
                f"{cond['cond']} → result config", ids, index=idx, key=f"sf_cfg_{i}",
                format_func=lambda cid: f"{cid}  ({by_id[cid]['hypothesis']} · {by_id[cid]['retriever']})",
            )
            entry = by_id[chosen]
            conf.update({"config_id": chosen, "result_file": entry["result_file"],
                         "hypothesis": entry["hypothesis"], "retriever": entry["retriever"]})
        st.caption(f"↳ `{conf['result_file']}`")


def _render_profiles(cfg: dict) -> None:
    st.subheader("Profiles (evaluation questions)")
    st.caption("Each profile is an evaluation question shown to participants. Defaults are diverse "
               "and not 'too easy' (many constraints, few valid recipes).")

    # Seed the editable list from the effective profiles when the admin hasn't set any yet.
    profiles = cfg.get("study_profiles")
    if not profiles:
        profiles = [dict(p) for p in get_study_profiles()]
        cfg["study_profiles"] = profiles

    files = [c["config"]["result_file"] for c in cfg.get("study_conditions", [])
             if c.get("config", {}).get("result_file")]

    ctop1, ctop2 = st.columns([1, 1])
    n_target = ctop1.number_input("Number of profiles", min_value=2, max_value=8,
                                  value=len(profiles), step=1, key="sf_nprof")
    if ctop2.button("✨ Suggest diverse defaults", key="sf_suggest",
                    help="Re-pick hard, diverse questions from the current conditions."):
        try:
            cfg["study_profiles"] = study_results.select_default_profiles(files, n=int(n_target))
            st.success(f"Selected {len(cfg['study_profiles'])} profiles. Review, save, then rebuild.")
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not select profiles: {exc}")

    remove_idx = None
    for i, prof in enumerate(profiles):
        with st.container(border=True):
            c1, c2, c3 = st.columns([1.2, 3.2, 0.6])
            with c1:
                prof["question_id"] = st.text_input("question_id", value=prof.get("question_id", ""),
                                                    key=f"sf_qid_{i}")
                prof.setdefault("id", f"P{i + 1}")
                st.caption(f"id: `{prof['id']}`")
            with c2:
                meta = _question_meta(prof.get("question_id", ""))
                if meta:
                    st.markdown(f"**{meta['goal']}** · {meta['constraint_count']} constraints · "
                                f"{meta['answer_set_size']} valid recipes")
                    st.caption(meta["prompt"][:220])
                else:
                    st.caption("Unknown question_id.")
            with c3:
                if st.button("🗑", key=f"sf_rm_{i}", help="Remove this profile"):
                    remove_idx = i
    if remove_idx is not None:
        profiles.pop(remove_idx)
        st.rerun()

    if st.button("＋ Add profile", key="sf_add"):
        profiles.append({"id": f"P{len(profiles) + 1}", "question_id": ""})
        st.rerun()


def _question_meta(qid: str) -> dict | None:
    if not qid:
        return None
    try:
        return study_results.profile_from_question(qid)
    except Exception:  # noqa: BLE001
        return None


def _render_status(cfg: dict) -> None:
    st.subheader("Plan artifacts")
    st.caption("Pre-generated by `uv run python scripts/build_study_plans.py`. "
               "Rebuild after changing conditions or profiles.")
    for prof in cfg.get("study_profiles", []):
        path: Path = STUDY_PLANS_DIR / f"{prof.get('id', '?')}.json"
        if path.exists():
            st.markdown(f"✅ `{prof.get('id')}` — {path.name}")
        else:
            st.markdown(f"⚠️ `{prof.get('id')}` — not built yet (`{path.name}` missing; "
                        f"study falls back to mock)")
    st.code("uv run python scripts/build_study_plans.py --select   # re-pick defaults\n"
            "uv run python scripts/build_study_plans.py            # rebuild plan artifacts",
            language="bash")


def render() -> None:
    cfg = get_admin_config()
    _persistence_controls(cfg)
    st.divider()
    _render_conditions(cfg)
    st.divider()
    _render_profiles(cfg)
    st.divider()
    _render_status(cfg)
