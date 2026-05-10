import csv
from functools import lru_cache
from pathlib import Path


DATASET_PATH = Path(__file__).resolve().parents[1] / "data" / "best_podcast_train.csv"
PODCAST_KEYWORDS = {
    "podcast",
    "podcasts",
    "recommend",
    "recommendation",
    "suggest",
    "suggestion",
    "listen",
    "episode",
    "episodes",
}

GENRE_ALIASES = {
    "tech": "Technology",
    "technology": "Technology",
    "true crime": "True Crime",
    "crime": "True Crime",
    "business": "Business",
    "comedy": "Comedy",
    "education": "Education",
    "health": "Health",
    "lifestyle": "Lifestyle",
    "music": "Music",
    "news": "News",
    "sports": "Sports",
}


def _normalize(text):
    return " ".join((text or "").strip().lower().split())


def _to_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@lru_cache(maxsize=1)
def _load_dataset():
    if not DATASET_PATH.exists():
        return []

    with DATASET_PATH.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    for row in rows:
        row["Listening_Time_minutes"] = _to_float(row.get("Listening_Time_minutes"), 0.0)
        row["Host_Popularity_percentage"] = _to_float(row.get("Host_Popularity_percentage"), 0.0)
        row["Episode_Sentiment"] = _to_float(row.get("Episode_Sentiment"), 0.0)
        row["Genre"] = (row.get("Genre") or "").strip()
        row["Publication_Day"] = (row.get("Publication_Day") or "").strip()
        row["Publication_Time"] = (row.get("Publication_Time") or "").strip()
        row["Podcast_Name"] = (row.get("Podcast_Name") or "Unknown Podcast").strip()
        row["Episode_Title"] = (row.get("Episode_Title") or "Unknown Episode").strip()
    return rows


def is_podcast_request(user_input):
    text = _normalize(user_input)
    return any(keyword in text for keyword in PODCAST_KEYWORDS)


def _extract_preferences(user_input):
    text = _normalize(user_input)

    genre = None
    # Prefer longest alias match first, so "true crime" wins before "crime".
    for alias, canonical in sorted(GENRE_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        if alias in text:
            genre = canonical
            break

    day = None
    for candidate in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]:
        if candidate.lower() in text:
            day = candidate
            break

    pub_time = None
    for candidate in ["Morning", "Afternoon", "Evening", "Night"]:
        if candidate.lower() in text:
            pub_time = candidate
            break

    return {"genre": genre, "day": day, "time": pub_time}


def _filter_rows(rows, preferences):
    filtered = rows
    if preferences.get("genre"):
        filtered = [r for r in filtered if r.get("Genre") == preferences["genre"]]
    if preferences.get("day"):
        filtered = [r for r in filtered if r.get("Publication_Day") == preferences["day"]]
    if preferences.get("time"):
        filtered = [r for r in filtered if r.get("Publication_Time") == preferences["time"]]
    return filtered


def _calculate_score(row):
    listening_time = row.get("Listening_Time_minutes", 0.0)
    host_pop = row.get("Host_Popularity_percentage", 0.0) / 100.0
    sentiment = (row.get("Episode_Sentiment", 0.0) + 1.0) / 2.0

    weighted_score = (
        listening_time * 0.50 +
        (host_pop * 100) * 0.30 +
        (sentiment * 100) * 0.20
    )
    return weighted_score


def recommend_podcasts_from_query(user_input, top_k=3):
    rows = _load_dataset()
    if not rows:
        return {"items": [], "preferences": {}, "mode": "dataset_missing"}

    preferences = _extract_preferences(user_input)
    filtered = _filter_rows(rows, preferences)

    # If strict filtering is too narrow, relax while preserving at least one requested preference.
    if not filtered and any(preferences.values()):
        filtered = rows
        if preferences.get("genre"):
            filtered = [r for r in filtered if r.get("Genre") == preferences["genre"]]
        elif preferences.get("day"):
            filtered = [r for r in filtered if r.get("Publication_Day") == preferences["day"]]
        elif preferences.get("time"):
            filtered = [r for r in filtered if r.get("Publication_Time") == preferences["time"]]

    if not filtered:
        filtered = rows

    ranked = sorted(filtered, key=_calculate_score, reverse=True)

    unique_items = []
    seen_podcasts = set()
    for row in ranked:
        podcast_name = row.get("Podcast_Name")
        if podcast_name in seen_podcasts:
            continue
        seen_podcasts.add(podcast_name)
        unique_items.append(
            {
                "podcast": podcast_name,
                "episode": row.get("Episode_Title"),
                "genre": row.get("Genre"),
                "day": row.get("Publication_Day"),
                "time": row.get("Publication_Time"),
                "score": _calculate_score(row),
            }
        )
        if len(unique_items) >= top_k:
            break

    mode = "filtered" if any(preferences.values()) else "trending"
    return {"items": unique_items, "preferences": preferences, "mode": mode}


def get_filter_options():
    """Return available options for each filter category."""
    rows = _load_dataset()
    if not rows:
        return {"genres": [], "days": [], "times": []}

    genres = [g for g in set(r.get("Genre") for r in rows) if g]
    genres = sorted(genres)
    
    days_set = [d for d in set(r.get("Publication_Day") for r in rows) if d]
    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    days = sorted(days_set, key=lambda x: days_order.index(x) if x in days_order else 999)
    
    times_set = [t for t in set(r.get("Publication_Time") for r in rows) if t]
    times_order = ["Morning", "Afternoon", "Evening", "Night"]
    times = sorted(times_set, key=lambda x: times_order.index(x) if x in times_order else 999)

    return {"genres": genres, "days": days, "times": times}
