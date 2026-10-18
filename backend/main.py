from fastapi import FastAPI, HTTPException, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os

from services import (
    process_players, get_seasons, get_player_trend, get_matches, generate_ai_summary
)
import insights
import replay

app = FastAPI(title="CricImpact API")


@app.on_event("startup")
def warm_caches():
    """Load the CSVs and featured replays in the background so the first page load is instant."""
    import threading

    def _warm():
        try:
            insights.overview()
            insights.player_list()
            replay.featured()
        except Exception as e:  # never block startup on a missing file
            print(f"Cache warm-up skipped: {e}")

    threading.Thread(target=_warm, daemon=True).start()

# CORS: in production set ALLOWED_ORIGINS to your frontend URL(s), comma-separated,
# e.g. https://cric-impact.vercel.app. Defaults to allowing any origin (local dev).
_origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_origin_regex=os.getenv("ALLOWED_ORIGIN_REGEX") or None,  # e.g. https://.*\.vercel\.app for previews
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "ok", "message": "CricImpact API is running"}

@app.get("/api/seasons")
def api_get_seasons():
    seasons = get_seasons()
    return {"seasons": seasons, "data": seasons}

@app.get("/api/players")
def api_get_players(
    min_inn: int = 10,
    role_pick: str = "All",
    season: str = "All",
    min_runs: int = 0,
    min_wkts: int = 0
):
    try:
        players = process_players(min_inn, role_pick, season, min_runs, min_wkts)
        return {"data": players}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/overview")
def api_overview():
    return {"data": insights.overview()}

@app.get("/api/player-list")
def api_player_list():
    return {"data": insights.player_list()}

@app.get("/api/compare")
def api_compare(a: str, b: str):
    try:
        return {"data": insights.compare(a, b)}
    except KeyError as e:
        raise HTTPException(status_code=404, detail=f"Player not found: {e}")

@app.get("/api/story")
def api_story():
    """Model card, featured thrillers and the runs-vs-impact insight for the home page."""
    return {"data": insights.clean({
        "model": replay.model_card(),
        "featured": replay.featured(),
        "runs_vs_impact": replay.runs_vs_impact(),
    })}

@app.get("/api/matches/{match_id}/worm")
def api_worm(match_id: int):
    try:
        return {"data": replay.worm(match_id)}
    except KeyError:
        raise HTTPException(status_code=404, detail="No ball-by-ball data for this match")
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

@app.get("/api/matches/{match_id}")
def api_match_detail(match_id: int):
    try:
        return {"data": insights.match_detail(match_id)}
    except KeyError:
        raise HTTPException(status_code=404, detail="Match not found")

@app.post("/api/upload")
async def api_upload(ball_file: UploadFile, info_file: UploadFile):
    import tempfile
    from match_impact_from_files import compute_match_rankings_from_files
    tmpdir = tempfile.mkdtemp()
    base = os.path.splitext(os.path.basename(ball_file.filename or "upload.csv"))[0]
    ball_path = os.path.join(tmpdir, f"{base}.csv")
    info_path = os.path.join(tmpdir, f"{base}_info.csv")
    with open(ball_path, "wb") as f:
        f.write(await ball_file.read())
    with open(info_path, "wb") as f:
        f.write(await info_file.read())
    try:
        from services import model_dir
        rankings = compute_match_rankings_from_files(ball_path, info_path, model_dir=model_dir())
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to compute impact: {e}")
    summary = insights.summarize_match(rankings.sort_values("rank"))
    mid = rankings["match_id"].iloc[0] if len(rankings) else base
    return {"data": insights.clean({"match_id": str(mid), **summary})}

@app.get("/api/players/{player_name}")
def api_player_detail(player_name: str):
    try:
        return {"data": insights.player_detail(player_name)}
    except KeyError:
        raise HTTPException(status_code=404, detail="Player not found")

@app.get("/api/players/{player_name}/trend")
def api_get_player_trend(player_name: str, season: Optional[str] = "All"):
    try:
        trend = get_player_trend(player_name, season)
        return {"data": trend}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/matches")
def api_get_matches(season: str = "All", team: str = "All"):
    try:
        matches = get_matches(season, team)
        return {"data": insights.clean(matches)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class SummaryRequest(BaseModel):
    match_id: str
    players: List[Dict[str, Any]]

@app.post("/api/generate_summary")
def api_generate_summary(request: SummaryRequest):
    import dotenv
    dotenv.load_dotenv(override=True)
    api_key = os.getenv("GROQ_API_KEY")
    
    if not api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY environment variable is not set")
        
    context = {"match_id": request.match_id, "players": request.players}
    result = generate_ai_summary(context, api_key)
    
    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("error"))
        
    return {"data": result}
