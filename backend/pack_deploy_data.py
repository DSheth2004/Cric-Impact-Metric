"""
Bundle what the backend reads into backend/data/ so it can be committed and deployed
(e.g. to Render), without the raw Cricsheet folder or the full pipeline outputs.

    python run_pipeline.py --data_folder ./ipl_male_csv2    # if outputs are stale
    python backend/pack_deploy_data.py

Re-run it whenever the pipeline output changes, then commit backend/data/.
"""
import gzip
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
SRC = os.path.join(ROOT, "cricket_impact")
OUT = os.path.join(ROOT, "backend", "data")

CSV_FILES = [
    "all_players_impact_T20.csv",
    "match_rankings.csv",
    "batting_impact.csv",
    "bowling_impact.csv",
    "batter_vs_bowler_rivalry.csv",
    "batting_partnerships.csv",
    "player_match_context.csv",
    "ball_win_prob.csv",
]


def gz(src: str, dst: str) -> None:
    with open(src, "rb") as f_in, gzip.open(dst, "wb", compresslevel=9) as f_out:
        shutil.copyfileobj(f_in, f_out)


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    total = 0
    for name in CSV_FILES:
        src = os.path.join(SRC, name)
        if not os.path.exists(src):
            raise SystemExit(f"Missing {src}. Run run_pipeline.py first.")
        dst = os.path.join(OUT, name + ".gz")
        gz(src, dst)
        total += os.path.getsize(dst)
        print(f"  {name:32s} {os.path.getsize(src) / 1e6:6.1f} MB -> {os.path.getsize(dst) / 1e6:5.1f} MB")

    for name, src in [("model_card.json", os.path.join(SRC, "model_card.json")),
                      ("win_prob_model.pkl", os.path.join(SRC, "models", "win_prob_model.pkl"))]:
        shutil.copy2(src, os.path.join(OUT, name))
        total += os.path.getsize(src)
        print(f"  {name:32s} copied")

    # winners, margins, venues and Player of the Match: replaces the raw 47 MB match folder
    from cricket_impact.data_loader import load_match_info
    info = load_match_info(os.path.join(ROOT, "ipl_male_csv2"))
    cols = ["match_id", "season", "start_date", "venue", "city", "winner", "winner_runs",
            "winner_wickets", "player_of_match", "match_number", "is_playoff"]
    dst = os.path.join(OUT, "match_info.csv.gz")
    info[[c for c in cols if c in info.columns]].to_csv(dst, index=False, compression="gzip")
    total += os.path.getsize(dst)
    print(f"  {'match_info.csv':32s} {len(info)} matches")
    print(f"Wrote backend/data/ ({total / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
