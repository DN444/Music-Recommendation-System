import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


class TfidfRecommender:
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)

        required = ["artist", "song", "link", "text"]
        for col in required:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column: {col}")

        for col in required:
            self.df[col] = self.df[col].fillna("").astype(str)

        self.df["artist_l"] = self.df["artist"].str.strip().str.lower()
        self.df["song_l"] = self.df["song"].str.strip().str.lower()

        self.df["combined"] = (
            self.df["artist"] + " " +
            self.df["song"] + " " +
            self.df["text"]
        ).str.lower()

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=80_000
        )
        self.tfidf = self.vectorizer.fit_transform(self.df["combined"])

        self.song_to_indices = {}
        for i, s in enumerate(self.df["song_l"]):
            self.song_to_indices.setdefault(s, []).append(i)

        self.song_artist_to_idx = {}
        for i, (s, a) in enumerate(zip(self.df["song_l"], self.df["artist_l"])):
            self.song_artist_to_idx.setdefault((s, a), i)

    def _resolve_index(self, song_name: str, artist: str | None = None) -> int:
        s = song_name.strip().lower()

        if artist and artist.strip():
            a = artist.strip().lower()
            key = (s, a)
            if key not in self.song_artist_to_idx:
                raise KeyError(f"Song+artist not found: {song_name} — {artist}")
            return self.song_artist_to_idx[key]

        if s not in self.song_to_indices:
            raise KeyError(f"Song not found: {song_name}")

        return self.song_to_indices[s][0]

    def recommend(self, song_name: str, k: int = 10, artist: str | None = None):
        if not song_name or not song_name.strip():
            raise ValueError("song_name cannot be empty")

        idx = self._resolve_index(song_name, artist=artist)

        sims = linear_kernel(self.tfidf[idx], self.tfidf).ravel()
        ranked = sims.argsort()[::-1]
        ranked = [i for i in ranked if i != idx][:k]

        recs = [{
            "song": self.df.at[i, "song"],
            "artist": self.df.at[i, "artist"],
            "link": self.df.at[i, "link"],
            "score": float(sims[i]),
        } for i in ranked]

        return {
            "query": {
                "song": self.df.at[idx, "song"],
                "artist": self.df.at[idx, "artist"],
            },
            "recommendations": recs
        }
