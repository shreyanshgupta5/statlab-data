#!/usr/bin/env python3
"""
Stat Lab — São Paulo GP 2026 practice (FP1 + FP2) from OpenF1 -> CSV files only.

Output (./brazil_practice_out/):
  fp1_classification.csv, fp2_classification.csv   best lap, gap, sectors, speed traps, tyre, laps
  fp1_all_laps.csv, fp2_all_laps.csv               every lap with sectors, speeds and tyre
  fp1_long_runs.csv, fp2_long_runs.csv             runs of 5+ clean laps: median pace, tyre wear (s/lap)
  fp1_weather.csv, fp2_weather.csv                 air/track temp, humidity, wind, rain through the session
  fp1_fastest_lap_telemetry_XXX.csv (x2 per session) speed, throttle, brake, gear on the two fastest laps
  practice_team_order.csv                          best lap and best long run per team

Run:  pip install requests pandas numpy
      python brazil_practice_data.py
      python brazil_practice_data.py --no-telemetry   # skip the fastest-lap telemetry files
      python brazil_practice_data.py --cache          # keep downloads so re-runs are instant

Every step prints how long it took.
"""
import argparse
import hashlib
import json
from contextlib import contextmanager
from urllib.parse import urlencode
import os
import sys
import time

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------- settings
YEAR, COUNTRY = 2026, "Brazil"
SESSIONS = ["Practice 1", "Practice 2"]          # add "Practice 3", "Qualifying" later (run with --fresh)
INTERLAGOS = (-23.7036, -46.6997)                 # for the weather archive
LAST_N_RACES = 5                                  # races shown on the "who wins here" slide

try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:                                 # notebooks have no __file__
    HERE = os.getcwd()
OUT = os.path.join(HERE, "brazil_practice_out")
os.makedirs(OUT, exist_ok=True)



def team_key(name):
    """Map any team or constructor name (current or historical) to today's team name."""
    n = (name or "").lower().replace("_", " ")
    for k, v in [("mercedes", "Mercedes"), ("ferrari", "Ferrari"), ("racing bulls", "Racing Bulls"),
                 ("alphatauri", "Racing Bulls"), ("toro rosso", "Racing Bulls"), ("rb f1", "Racing Bulls"),
                 ("visa cash", "Racing Bulls"), ("red bull", "Red Bull"), ("mclaren", "McLaren"),
                 ("aston", "Aston Martin"), ("racing point", "Aston Martin"), ("force india", "Aston Martin"),
                 ("alpine", "Alpine"), ("renault", "Alpine"), ("williams", "Williams"), ("haas", "Haas"),
                 ("audi", "Audi"), ("sauber", "Audi"), ("alfa", "Audi"), ("cadillac", "Cadillac")]:
        if k in n:
            return v
    if n.strip() == "rb":
        return "Racing Bulls"
    return name




# ---------------------------------------------------------------- fast, cached HTTP
CACHE = os.path.join(HERE, "cache")
USE_CACHE = False                                  # turn on with --cache
HTTP = requests.Session()
HTTP.headers["User-Agent"] = "statlab/1.0"


@contextmanager
def step(label):
    """Prints how long each step takes, so you can see where time goes."""
    t = time.time()
    print(f"  · {label} …", flush=True)
    yield
    print(f"    done in {time.time() - t:.1f}s", flush=True)


def _get(url, params=None, tries=3):
    """GET with a disk cache, short timeouts and polite retries. Returns parsed JSON or None."""
    key = url + "?" + urlencode(sorted((params or {}).items()))
    path = os.path.join(CACHE, hashlib.md5(key.encode()).hexdigest() + ".json")
    if USE_CACHE and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    for attempt in range(tries):
        try:
            r = HTTP.get(url, params=params, timeout=(8, 30))   # 8 s to connect, 30 s to answer
        except requests.RequestException as e:
            print(f"      {type(e).__name__} on {url.split('/')[2]}, retry {attempt + 1}/{tries}", flush=True)
            time.sleep(2 * (attempt + 1)); continue
        if r.status_code == 200:
            data = r.json()
            if USE_CACHE:
                os.makedirs(CACHE, exist_ok=True)
                with open(path, "w") as f:
                    json.dump(data, f)
            return data
        if r.status_code == 429:
            wait = min(int(r.headers.get("Retry-After", 5) or 5), 20)
            print(f"      rate-limited by {url.split('/')[2]}, waiting {wait}s", flush=True)
            time.sleep(wait); continue
        if r.status_code >= 500:
            time.sleep(3); continue
        if r.status_code == 401 and "openf1" in url:
            print("      OpenF1 refuses requests while a session is live; run again after it ends", flush=True)
        return None
    print(f"      gave up on {url}", flush=True)
    return None


def openf1(endpoint, raw="", **params):
    """raw: extra query text sent as-is, for OpenF1's date filters like date>=..."""
    time.sleep(0.3)
    url = f"https://api.openf1.org/v1/{endpoint}" + (f"?{raw}" if raw else "")
    return _get(url, params) or []


def jolpica(path):
    d = _get(f"https://api.jolpi.ca/ergast/f1/{path}.json", {"limit": 100})
    time.sleep(0.3)
    return d["MRData"] if d else {}


def open_meteo(date, hours):
    """Hourly race-window weather at Interlagos from the free Open-Meteo archive."""
    d = _get("https://archive-api.open-meteo.com/v1/archive",
             {"latitude": INTERLAGOS[0], "longitude": INTERLAGOS[1], "start_date": date, "end_date": date,
              "hourly": "temperature_2m,precipitation,relative_humidity_2m,wind_speed_10m",
              "timezone": "America/Sao_Paulo"})
    if not d or "hourly" not in d:
        return {}
    h = pd.DataFrame(d["hourly"])
    h["hour"] = pd.to_datetime(h["time"]).dt.hour
    w = h[h["hour"].isin(hours)]
    if w.empty:
        return {}
    return {"air_c": round(w["temperature_2m"].mean(), 1), "rain_mm": round(w["precipitation"].sum(), 1),
            "humidity": round(w["relative_humidity_2m"].mean()), "wind_kmh": round(w["wind_speed_10m"].max(), 1)}


def lap_str(s):
    if s is None or pd.isna(s):
        return "–"
    m = int(s // 60)
    return f"{m}:{s - 60 * m:06.3f}"


# ================================================================ PART A · practice
def practice_session(name, telemetry=True):
    sess = openf1("sessions", year=YEAR, country_name=COUNTRY, session_name=name)
    if not sess:
        print(f"  {name}: not on OpenF1 yet"); return None
    s = sess[-1]; sk = s["session_key"]
    print(f"    session {sk}: downloading laps, stints, weather, race control", flush=True)
    drivers = {d["driver_number"]: d for d in openf1("drivers", session_key=sk)}
    laps = pd.DataFrame(openf1("laps", session_key=sk))
    if laps.empty:
        print(f"  {name}: no laps yet"); return None
    stints = pd.DataFrame(openf1("stints", session_key=sk))
    weather = pd.DataFrame(openf1("weather", session_key=sk))
    rc = pd.DataFrame(openf1("race_control", session_key=sk))
    laps["driver"] = laps["driver_number"].map(lambda n: drivers.get(n, {}).get("name_acronym", str(n)))
    laps["team"] = laps["driver_number"].map(lambda n: team_key(drivers.get(n, {}).get("team_name", "")))
    # tyre per lap from the stints table
    laps["compound"] = None
    if not stints.empty:
        for _, st in stints.iterrows():
            m = (laps["driver_number"] == st["driver_number"]) & (laps["lap_number"] >= st["lap_start"]) & \
                (laps["lap_number"] <= (st["lap_end"] if pd.notna(st.get("lap_end")) else 999))
            laps.loc[m, "compound"] = st.get("compound")
            laps.loc[m, "stint"] = st.get("stint_number")
    timed = laps[laps["lap_duration"].notna() & (laps["is_pit_out_lap"] != True)]

    # classification
    rows = []
    for dn, g in laps.groupby("driver_number"):
        t = timed[timed["driver_number"] == dn]
        b = t.loc[t["lap_duration"].idxmin()] if len(t) else None
        rows.append({"session": name, "driver": g["driver"].iloc[0], "team": g["team"].iloc[0], "number": dn,
                     "best": None if b is None else round(float(b["lap_duration"]), 3),
                     "s1": None if b is None else b.get("duration_sector_1"), "s2": None if b is None else b.get("duration_sector_2"),
                     "s3": None if b is None else b.get("duration_sector_3"),
                     "tyre": None if b is None else b.get("compound"), "best_lap_no": None if b is None else int(b["lap_number"]),
                     "laps": int(g["lap_number"].max()),
                     "i1_speed_max": g["i1_speed"].max() if "i1_speed" in g else None,
                     "i2_speed_max": g["i2_speed"].max() if "i2_speed" in g else None,
                     "speed_trap_max": g["st_speed"].max() if "st_speed" in g else None})
    cls = pd.DataFrame(rows).sort_values("best", na_position="last").reset_index(drop=True)
    cls.insert(0, "pos", range(1, len(cls) + 1))
    cls["gap"] = (cls["best"] - cls["best"].iloc[0]).round(3)
    ideal = cls[["s1", "s2", "s3"]].min()             # best sectors of the session (theoretical best lap)
    cls["ideal_lap"] = (cls[["s1", "s2", "s3"]].sum(axis=1, min_count=3)).round(3)

    # long runs: stints with 5+ clean laps; drop laps >107% of the stint median (traffic, cool-down)
    lr = []
    for (dn, stn), g in timed.dropna(subset=["stint"]).groupby(["driver_number", "stint"]):
        g = g.sort_values("lap_number").iloc[1:]                    # first flying lap of a stint is often a push or warm-up
        if len(g) < 5:
            continue
        med = g["lap_duration"].median()
        clean = g[g["lap_duration"] < med * 1.07]
        if len(clean) < 5:
            continue
        slope = np.polyfit(np.arange(len(clean)), clean["lap_duration"].to_numpy(), 1)[0]
        lr.append({"session": name, "driver": g["driver"].iloc[0], "team": g["team"].iloc[0], "stint": int(stn),
                   "compound": g["compound"].iloc[0], "laps": len(clean), "median": round(clean["lap_duration"].median(), 3),
                   "mean": round(clean["lap_duration"].mean(), 3), "deg_s_per_lap": round(float(slope), 3)})
    longruns = pd.DataFrame(lr)

    # weather summary
    wx = {}
    if not weather.empty:
        wx = {"air_min": weather["air_temperature"].min(), "air_max": weather["air_temperature"].max(),
              "track_min": weather["track_temperature"].min(), "track_max": weather["track_temperature"].max(),
              "humidity": round(weather["humidity"].mean()), "wind_max": weather["wind_speed"].max(),
              "rain": bool(weather["rainfall"].astype(float).max() > 0)}
    # race control
    rcs = {}
    if not rc.empty:
        msg = rc["message"].fillna("").str.upper()
        rcs = {"red_flags": int((rc.get("flag", pd.Series(dtype=str)) == "RED").sum()),
               "yellow_flags": int(rc.get("flag", pd.Series(dtype=str)).isin(["YELLOW", "DOUBLE YELLOW"]).sum()),
               "deleted_laps": int(msg.str.contains("DELETED").sum()),
               "investigations": int(msg.str.contains("NOTED|UNDER INVESTIGATION", regex=True).sum())}
    # speed trace of the two fastest laps
    traces = {}
    for _, r in (cls.dropna(subset=["best"]).head(2).iterrows() if telemetry else []):
        lap = timed[(timed["driver"] == r["driver"]) & (timed["lap_number"] == r["best_lap_no"])].iloc[0]
        start = pd.to_datetime(lap["date_start"], utc=True, format="ISO8601")
        end = start + pd.Timedelta(seconds=float(lap["lap_duration"]))
        fmt = "%Y-%m-%dT%H:%M:%S.%f"
        car = pd.DataFrame(openf1("car_data", raw=f"date>={start.strftime(fmt)}&date<={end.strftime(fmt)}",
                                  session_key=sk, driver_number=int(lap["driver_number"])))
        if len(car) > 50:
            car["t"] = (pd.to_datetime(car["date"], utc=True, format="ISO8601") - start).dt.total_seconds()
            v, t = car["speed"].to_numpy() / 3.6, car["t"].to_numpy()
            car["dist"] = np.r_[0, np.cumsum(np.diff(t) * (v[1:] + v[:-1]) / 2)]
            traces[r["driver"]] = car[["dist", "speed", "throttle", "brake", "n_gear"]]
    return {"name": name, "session": s, "cls": cls, "longruns": longruns, "laps": laps, "weather": wx,
            "weather_raw": weather, "race_control": rcs, "traces": traces, "ideal": ideal}


def team_table(practice):
    rows = []
    for p in practice:
        for team, g in p["cls"].dropna(subset=["best"]).groupby("team"):
            rows.append({"session": p["name"], "team": team, "best": g["best"].min()})
        if not p["longruns"].empty:
            for team, g in p["longruns"].groupby("team"):
                rows.append({"session": p["name"], "team": team, "longrun": g["median"].min()})
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    t = t.groupby("team").agg(best=("best", "min"), longrun=("longrun", "min")).reset_index()
    t["gap_best"] = (t["best"] - t["best"].min()).round(3)
    t["gap_longrun"] = (t["longrun"] - t["longrun"].min()).round(3)
    return t.sort_values("best")


# ================================================================ main
def main():
    global USE_CACHE
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", action="store_true", help="save downloads in ./cache so re-runs are instant")
    ap.add_argument("--no-telemetry", action="store_true", help="skip the speed-trace download")
    args, _ = ap.parse_known_args()
    USE_CACHE = args.cache
    t0 = time.time()
    practice = []
    for name in SESSIONS:
        with step(name):
            p = practice_session(name, telemetry=not args.no_telemetry)
        if p:
            practice.append(p)
    if not practice:
        sys.exit("No practice data found yet.")
    for p in practice:
        tag = p["name"].replace("Practice ", "fp")
        p["cls"].to_csv(os.path.join(OUT, f"{tag}_classification.csv"), index=False)
        p["laps"].to_csv(os.path.join(OUT, f"{tag}_all_laps.csv"), index=False)
        if not p["longruns"].empty:
            p["longruns"].to_csv(os.path.join(OUT, f"{tag}_long_runs.csv"), index=False)
        if not p["weather_raw"].empty:
            p["weather_raw"].to_csv(os.path.join(OUT, f"{tag}_weather.csv"), index=False)
        for drv, tr in p["traces"].items():
            tr.to_csv(os.path.join(OUT, f"{tag}_fastest_lap_telemetry_{drv}.csv"), index=False)
        print(f"\n  {p['name']} top 5:\n" + p["cls"][["pos", "driver", "team", "best", "gap", "tyre", "laps"]].head(5).to_string(index=False))
        wx = {k: (round(float(v), 1) if isinstance(v, (int, float, np.floating)) and not isinstance(v, bool) else v) for k, v in p["weather"].items()}
        print(f"  weather {wx} · race control {p['race_control']}")
    tt = team_table(practice)
    if not tt.empty:
        tt.to_csv(os.path.join(OUT, "practice_team_order.csv"), index=False)
    print(f"\nDone in {time.time() - t0:.0f}s: CSVs saved in {OUT}")


if __name__ == "__main__":
    main()
