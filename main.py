import os
from contextlib import asynccontextmanager

import gdown
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse

from recommender import TfidfRecommender

DEFAULT_DRIVE_URL = "https://drive.google.com/file/d/18Xw1zxYT78mryprcHMoNwKKiMePrk2h3/view?usp=sharing"

DATASET_URL = os.getenv("DATASET_URL", DEFAULT_DRIVE_URL)

CSV_PATH = os.getenv("CSV_PATH", "spotify_millsongdata.csv")

def download_if_missing(url: str, path: str):
    if os.path.exists(path):
        return

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    # fuzzy=True allows using the normal "file/d/<id>/view" sharing URL
    out = gdown.download(url=url, output=path, quiet=False, fuzzy=True)
    if not out or not os.path.exists(path):
        raise RuntimeError(
            "Download failed. Ensure the Google Drive file is set to 'Anyone with the link'."
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    download_if_missing(DATASET_URL, CSV_PATH)
    app.state.reco = TfidfRecommender(CSV_PATH)
    yield


app = FastAPI(title="Music Recommender (TF-IDF + Cosine)", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def home():
    html = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Music Recommender</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 40px; max-width: 900px; }
    input, button { padding: 10px; font-size: 14px; }
    input { width: 520px; }
    .row { margin: 12px 0; }
    .item { margin: 10px 0; padding: 10px; border: 1px solid #ddd; border-radius: 8px; }
    .muted { color: #666; }
  </style>
</head>
<body>
  <h2>Music recommendations</h2>

  <div class="row">
    <label>Song title:</label><br/>
    <input id="song" placeholder="Enter a song title from the dataset" />
  </div>

  <div class="row">
    <label>Artist (optional, for duplicates):</label><br/>
    <input id="artist" placeholder="Optional artist name" />
  </div>

  <div class="row">
    <label>How many (k):</label><br/>
    <input id="k" type="number" value="10" min="1" max="50" />
  </div>

  <button onclick="getRecs()">Get recommendations</button>

  <p id="status" class="muted"></p>
  <div id="results"></div>

<script>
async function getRecs() {
  const song = document.getElementById("song").value.trim();
  const artist = document.getElementById("artist").value.trim();
  const k = document.getElementById("k").value;

  const status = document.getElementById("status");
  const results = document.getElementById("results");
  results.innerHTML = "";

  if (!song) {
    status.textContent = "Please enter a song title.";
    return;
  }

  const params = new URLSearchParams({ song, k });
  if (artist) params.set("artist", artist);

  status.textContent = "Loading...";
  try {
    const r = await fetch("/recommend?" + params.toString());
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || "Request failed");

    status.textContent =
      `Found ${data.recommendations.length} recommendations for: ${data.query.song} — ${data.query.artist}`;

    data.recommendations.forEach(rec => {
      const div = document.createElement("div");
      div.className = "item";
      const linkHtml = rec.link ? `<a href="${rec.link}" target="_blank" rel="noreferrer">Link</a>` : "";
      div.innerHTML = `
        <div><b>${rec.song}</b> — ${rec.artist}</div>
        <div class="muted">Score: ${rec.score.toFixed(4)} ${linkHtml ? " | " + linkHtml : ""}</div>
      `;
      results.appendChild(div);
    });
  } catch (e) {
    status.textContent = "Error: " + e.message;
  }
}
</script>
</body>
</html>
"""
    return HTMLResponse(content=html)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/recommend")
def recommend(
    request: Request,
    song: str = Query(..., description="Song title from the dataset"),
    artist: str | None = Query(None, description="Optional artist to disambiguate duplicates"),
    k: int = Query(10, ge=1, le=50, description="Number of recommendations (1-50)")
):
    try:
        return request.app.state.reco.recommend(song_name=song, artist=artist, k=k)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))