from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from model import predict_match, df

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MatchRequest(BaseModel):
    home_team: str
    away_team: str


# ROOT
@app.get("/")
def root():
    return {"message": "Football Quant Lab API is running"}


# ENDPOINT TEAMS
@app.get("/teams")
def get_teams():
    teams = sorted(df["HomeTeam"].unique().tolist())
    return {"teams": teams}


# ENDPOINT PREDICT
@app.post("/predict")
def predict(match: MatchRequest):

    result = predict_match(match.home_team, match.away_team)

    if result is None:
        return {"error": "Not enough historical data for one of the teams"}

    return result