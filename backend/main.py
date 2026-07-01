from fastapi import FastAPI, HTTPException, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os

from services import (
    process_players, get_seasons, get_player_trend, get_matches, generate_ai_summary
)

app = FastAPI(title="CricImpact API")

# Allow CORS for React frontend (default dev port 5173 and others)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For development, allow all
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "ok", "message": "CricImpact API is running"}

@app.get("/api/seasons")
def api_get_seasons():
    return {"seasons": get_seasons()}

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
        return {"data": matches}
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
