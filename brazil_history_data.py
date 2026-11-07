#!/usr/bin/env python3
"""
Stat Lab — Interlagos history (Brazilian / São Paulo GPs) -> CSV files only.

Output (./brazil_history_out/):
  interlagos_races.csv        one row per race: winner, grid, pole, margin, podium, fastest lap,
                              retirements, biggest climber, race-day weather, safety cars (2023+)
  interlagos_results.csv      full classification of every race
  teams_at_interlagos.csv     wins, podiums, points per team (today's team names), last 5 races
  drivers_at_interlagos.csv   this season's drivers: races, wins, podiums, average finish, places gained

Run:  pip install requests pandas numpy
      python brazil_history_data.py                    # 2021–2025
      python brazil_history_data.py --years 2016 2025  # longer history (2020 had no race)
      python brazil_history_data.py --cache            # keep downloads so re-runs are instant

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
INTERLAGOS = (-23.7036, -46.6997)                 # for the weather archive
LAST_N_RACES = 5                                  # races shown on the "who wins here" slide

try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:                                 # notebooks have no __file__
    HERE = os.getcwd()
OUT = os.path.join(HERE, "brazil_history_out")
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


# ================================================================ PART B · history
def history(first, last):
    races, results = [], []
    for y in range(first, last + 1):
        ty = time.time()
        md = jolpica(f"{y}/circuits/interlagos/results").get("RaceTable", {}).get("Races", [])
        if not md:
            continue
        race = md[0]
        q = jolpica(f"{y}/circuits/interlagos/qualifying").get("RaceTable", {}).get("Races", [])
        pole = q[0]["QualifyingResults"][0]["Driver"].get("code") if q else None
        rows = []
        for r in race["Results"]:
            grid = int(r["grid"]) or None
            rows.append({"year": y, "pos": int(r["position"]), "classified": r["positionText"].isdigit(),
                         "code": r["Driver"].get("code") or r["Driver"]["familyName"][:3].upper(),
                         "driver": f"{r['Driver']['givenName']} {r['Driver']['familyName']}",
                         "team_hist": r["Constructor"]["name"], "team": team_key(r["Constructor"]["name"]),
                         "grid": grid, "points": float(r["points"]), "status": r["status"],
                         "time": r.get("Time", {}).get("time"), "fl_rank": int(r.get("FastestLap", {}).get("rank", 99))})
        res = pd.DataFrame(rows)
        n = len(res)
        res["grid_eff"] = res["grid"].fillna(n)
        res["gained"] = (res["grid_eff"] - res["pos"]).where(res["classified"])
        w, p2 = res.iloc[0], res.iloc[1]
        margin = None                                  # filled in by margins_fix()
        c = res.loc[res["gained"].idxmax()]
        # race-day weather: race start (UTC) -> local hours for the archive
        start = pd.to_datetime(f"{race['date']}T{race.get('time', '17:00:00Z')}").tz_convert("America/Sao_Paulo")
        wx = open_meteo(race["date"], [start.hour, (start.hour + 1) % 24])
        sc = {}
        if y >= 2023:
            sess = openf1("sessions", year=y, country_name="Brazil", session_name="Race")
            if sess:
                msgs = [(m.get("message") or "").upper() for m in openf1("race_control", session_key=sess[-1]["session_key"])]
                sc = {"safety_cars": sum("SAFETY CAR DEPLOYED" in m and "VIRTUAL" not in m for m in msgs),
                      "vscs": sum("VIRTUAL SAFETY CAR DEPLOYED" in m for m in msgs),
                      "red_flags": sum(m.startswith("RED FLAG") for m in msgs)}
        races.append({"year": y, "race": race["raceName"], "date": race["date"], "winner": w["code"], "winner_name": w["driver"],
                      "winner_team": w["team"], "winner_grid": int(w["grid_eff"]), "pitlane_start": pd.isna(w["grid"]),
                      "pole": pole, "pole_won": pole == w["code"], "margin_s": margin, "podium": " · ".join(res.iloc[:3]["code"]),
                      "fastest_lap": (res[res["fl_rank"] == 1]["code"].iloc[0] if (res["fl_rank"] == 1).any() else None),
                      "finishers": int(res["classified"].sum()), "dnfs": int((~res["classified"]).sum()),
                      "top_climber": c["code"], "climb": int(c["gained"]), "climb_from": int(c["grid_eff"]), "climb_to": int(c["pos"]),
                      **wx, **sc})
        results.append(res)
        print(f"    {y}: {w['code']} from P{int(w['grid_eff'])} · pole {pole} · {wx.get('air_c', '?')}°C · rain {wx.get('rain_mm', '?')} mm ({time.time() - ty:.1f}s)", flush=True)
    return pd.DataFrame(races), (pd.concat(results, ignore_index=True) if results else pd.DataFrame())


def margins_fix(races, results):
    """Winning margin from P2's '+x.xxx' or '+m:ss.sss' gap."""
    def secs(s):
        s = str(s).lstrip("+")
        parts = [float(p) for p in s.split(":")]
        return sum(p * 60 ** i for i, p in enumerate(reversed(parts)))
    out = []
    for y in races["year"]:
        p2 = results[(results["year"] == y) & (results["pos"] == 2)]
        t = p2["time"].iloc[0] if len(p2) else None
        out.append(round(secs(t), 3) if t and str(t).startswith("+") else None)
    races["margin_s"] = out
    return races


def team_history(results, years):
    r = results[results["year"].isin(years)]
    t = r.groupby("team").agg(points=("points", "sum"), wins=("pos", lambda x: int((x == 1).sum())),
                              podiums=("pos", lambda x: int((x <= 3).sum())),
                              avg_finish=("pos", "mean")).reset_index()
    t["avg_finish"] = t["avg_finish"].round(1)
    return t.sort_values(["points", "wins"], ascending=False)


def driver_history(results, current_codes):
    r = results[results["code"].isin(current_codes)]
    d = r.groupby("code").agg(starts=("year", "nunique"), wins=("pos", lambda x: int((x == 1).sum())),
                              podiums=("pos", lambda x: int((x <= 3).sum())),
                              avg_finish=("pos", "mean"), gained=("gained", "mean"),
                              best=("pos", "min")).reset_index()
    d["avg_finish"] = d["avg_finish"].round(1); d["gained"] = d["gained"].round(1)
    return d.sort_values(["wins", "podiums", "avg_finish"], ascending=[False, False, True])


def current_drivers():
    """Driver codes on this season's entry list (Jolpica)."""
    d = jolpica(f"{YEAR}/drivers").get("DriverTable", {}).get("Drivers", [])
    return {x.get("code") for x in d if x.get("code")}



# ================================================================ main
def main():
    global USE_CACHE
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs=2, type=int, default=[2021, 2025], help="history range, e.g. --years 2016 2025")
    ap.add_argument("--cache", action="store_true", help="save downloads in ./cache so re-runs are instant")
    args, _ = ap.parse_known_args()
    USE_CACHE = args.cache
    t0 = time.time()
    with step(f"races {args.years[0]}–{args.years[1]}"):
        races, results = history(*args.years)
    if races.empty:
        sys.exit("No Interlagos races found in that range.")
    races = margins_fix(races, results)
    races.to_csv(os.path.join(OUT, "interlagos_races.csv"), index=False)
    results.to_csv(os.path.join(OUT, "interlagos_results.csv"), index=False)
    years = sorted(races["year"])[-LAST_N_RACES:]
    th = team_history(results, years)
    with step("this season's drivers"):
        dh = driver_history(results, current_drivers())
    th.to_csv(os.path.join(OUT, "teams_at_interlagos.csv"), index=False)
    dh.to_csv(os.path.join(OUT, "drivers_at_interlagos.csv"), index=False)
    print("\n" + races[["year", "winner", "winner_grid", "pole", "margin_s", "air_c", "rain_mm"]].to_string(index=False))
    print(f"\nDone in {time.time() - t0:.0f}s: CSVs saved in {OUT}")


if __name__ == "__main__":
    main()
