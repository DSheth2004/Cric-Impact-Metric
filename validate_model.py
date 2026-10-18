"""
Compare the original win probability model (v1) with the improved one (v2).

    python validate_model.py --data_folder ./ipl_male_csv2

v1 = Random Forest, trained on 80% of matches and used to score every match
     (cricket_impact/models/win_prob_model_rf_v1.pkl, kept as a backup).
v2 = monotonic gradient boosting, every match scored out-of-fold (5-fold cross-fitting).

Reports, and writes into cricket_impact/model_card.json under "validation":
  1. Model accuracy   AUC and Brier, both as originally reported and under identical
                      5-fold cross-fitting (the fair comparison)
  2. Per-ball sanity  share of wickets that raise the batting side's win probability,
                      boundaries that lower it, and dot balls that raise it
  3. Player of the Match agreement (both models credited the same way, full match
     including the innings break)
                      how often the player with the most WPA in a match was the
                      official Player of the Match
"""
import argparse
import contextlib
import io
import json
import os
import sys
import tempfile
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, train_test_split

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from cricket_impact.data_loader import load_and_prepare  # noqa: E402
from cricket_impact.context_features import add_all_context_features  # noqa: E402
from cricket_impact.win_prob_model import (  # noqa: E402
    cross_fit_win_prob, predict_win_prob, prepare_training_data, make_model, V1_FEATURES,
)
from cricket_impact.wpa_engine import compute_per_ball_wpa  # noqa: E402

OUT = os.path.join(ROOT, "cricket_impact")
V1 = os.path.join(OUT, "models", "win_prob_model_rf_v1.pkl")


def sanity(b: pd.DataFrame) -> dict:
    w = b[b["is_wicket"] == 1]
    f = b[b["runs_off_bat"] >= 4]
    d = b[(b["is_wicket"] == 0) & (b["total_runs"] == 0)]
    early = b[(b["innings"] == 1) & (b["over"] < 10) & (b["total_runs"] == 0) & (b["is_wicket"] == 0)]
    return {
        "early_dots_moving_over_3pp": round(float((early["wpa"].abs() > 0.03).mean()), 4),
        "wickets_raising_batting_wp": round(float((w["wpa"] > 1e-9).mean()), 4),
        "boundaries_lowering_batting_wp": round(float((f["wpa"] < -1e-9).mean()), 4),
        "dots_raising_batting_wp": round(float((d["wpa"] > 1e-9).mean()), 4),
    }


def pom_agreement(b: pd.DataFrame, info: pd.DataFrame) -> dict:
    bat = b.groupby(["match_id", "striker"])["wpa"].sum().rename_axis(["match_id", "player"])
    bowl = (-b.groupby(["match_id", "bowler"])["wpa"].sum()).rename_axis(["match_id", "player"])
    tot = pd.concat([bat, bowl]).groupby(level=[0, 1]).sum().reset_index(name="wpa")
    top = tot.sort_values("wpa", ascending=False).groupby("match_id").head(1)
    top3 = tot.sort_values("wpa", ascending=False).groupby("match_id").head(3)
    pom = info[info["player_of_match"] != ""].set_index("match_id")["player_of_match"]
    top = top[top["match_id"].isin(pom.index)]
    hit1 = (top["player"].values == pom.reindex(top["match_id"]).values).mean()
    t3 = top3[top3["match_id"].isin(pom.index)]
    hit3 = t3.assign(p=pom.reindex(t3["match_id"]).values).query("player == p")["match_id"].nunique() / len(top)
    return {"matches": int(len(top)), "top_wpa_is_pom": round(float(hit1), 4), "pom_in_top3_wpa": round(float(hit3), 4)}


def main(data_folder: str) -> None:
    print("Loading data and features...")
    with contextlib.redirect_stdout(io.StringIO()):
        ball, info = load_and_prepare(data_folder)
        ball = add_all_context_features(ball, info)
    X, y, mids, _ = prepare_training_data(ball, info)
    X1, _, _, _ = prepare_training_data(ball, info, features=V1_FEATURES)   # same rows, v1 inputs
    folds = list(GroupKFold(n_splits=5).split(X, y, groups=mids))

    res = {"v1": {}, "v2": {}}

    # ── v1 as deployed ──
    if not os.path.exists(V1):
        raise SystemExit(f"Missing {V1}: the original model backup is needed for the comparison.")
    v1 = joblib.load(V1)
    _, test_m = train_test_split(np.unique(mids), test_size=0.2, random_state=42)
    held = np.isin(mids, test_m)
    p1 = v1.predict_proba(X1[held])[:, 1]
    res["v1"]["reported_auc"] = round(float(roc_auc_score(y[held], p1)), 4)
    res["v1"]["reported_brier"] = round(float(brier_score_loss(y[held], p1)), 4)
    p1all = predict_win_prob(ball, v1, features=V1_FEATURES)
    res["v1"]["per_ball"] = sanity(compute_per_ball_wpa(p1all))            # model behaviour only
    res["v1"]["player_of_match"] = pom_agreement(compute_per_ball_wpa(p1all, info), info)

    # ── fair accuracy: both model types under the same 5-fold cross-fitting ──
    print("Cross-fitting both model types (fair comparison)...")
    for key, fn, XX in [("v1", lambda: RandomForestClassifier(n_estimators=200, max_depth=10, n_jobs=-1, random_state=42), X1),
                        ("v2", make_model, X)]:
        p = np.zeros(len(y))
        for tr, te in folds:
            p[te] = fn().fit(XX[tr], y[tr]).predict_proba(XX[te])[:, 1]
        res[key]["crossfit_auc"] = round(float(roc_auc_score(y, p)), 4)
        res[key]["crossfit_brier"] = round(float(brier_score_loss(y, p)), 4)

    # ── v2 as deployed (out-of-fold everywhere) ──
    with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
        b2, _ = cross_fit_win_prob(ball, info, n_splits=5, model_dir=tmp)
    res["v2"]["per_ball"] = sanity(compute_per_ball_wpa(b2))
    res["v2"]["player_of_match"] = pom_agreement(compute_per_ball_wpa(b2, info), info)

    print(json.dumps(res, indent=2))
    path = os.path.join(OUT, "model_card.json")
    card = json.load(open(path)) if os.path.exists(path) else {}
    card["validation"] = res
    with open(path, "w") as f:
        json.dump(card, f, indent=2)
    print(f"Saved to {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_folder", default=os.path.join(ROOT, "ipl_male_csv2"))
    main(ap.parse_args().data_folder)
