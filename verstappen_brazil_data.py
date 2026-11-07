#!/usr/bin/env python3
"""
Stat Lab — Max Verstappen at Interlagos, lap by lap -> CSV files only.

Output (./verstappen_brazil_out/):
  ver_brazil_summary.csv      one row per race: team, grid, finish, places gained, laps led, best lap and its rank,
                              stops, compounds (2023+), on-track passes, winner, gap to winner, rain at race time
  ver_brazil_laps.csv         every Verstappen lap: position, lap time, gap to leader / car ahead / car behind, pit lap
  ver_brazil_passes.csv       every place he gained or lost and who it was against, with the cause:
                              start, on track, pit cycle (someone else stopped), own stop, retirement
  ver_brazil_pitstops.csv     his stops: lap, stationary time where available, positions before and after
  ver_brazil_stints.csv       tyre stints (OpenF1, 2023 onwards)
  ver_brazil_race_control.csv safety cars, VSCs, red flags and his penalties/investigations (OpenF1, 2023 onwards)
  ver_brazil_overtakes_openf1.csv  OpenF1's own overtake records for his races, if the API has them
  all_drivers_positions.csv   position of every car on every lap (for position-trace charts)

Run:  pip install requests pandas numpy
      python verstappen_brazil_data.py                    # 2015-2025 (2020 had no race)
      python verstappen_brazil_data.py --years 2016 2025
      python verstappen_brazil_data.py --cache            # keep downloads so re-runs are instant

Sources: Jolpica (Ergast successor) for results, lap times and pit stops; OpenF1 for tyres and
race control from 2023; Open-Meteo for rain. "On track" passes are inferred from lap-end order,
so a pass and re-pass inside one lap does not show, and the exact corner is not known.
"""
import argparse
import hashlib
import json
import os
import time
from contextlib import contextmanager
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import requests

DRIVER_ID, CODE = "max_verstappen", "VER"
INTERLAGOS = (-23.7036, -46.6997)

try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:                                 # notebooks have no __file__
    HERE = os.getcwd()
OUT = os.path.join(HERE, "verstappen_brazil_out")
CACHE = os.path.join(HERE, "cache")
USE_CACHE = False
HTTP = requests.Session()
HTTP.headers["User-Agent"] = "statlab/1.0"


@contextmanager
def step(label):
    t = time.time()
    print(f"  · {label} …", flush=True)
    yield
    print(f"    done in {time.time() - t:.1f}s", flush=True)


def _get(url, params=None, tries=3):
    """GET with optional disk cache, short timeouts and polite retries. Returns JSON or None."""
    key = url + "?" + urlencode(sorted((params or {}).items()))
    path = os.path.join(CACHE, hashlib.md5(key.encode()).hexdigest() + ".json")
    if USE_CACHE and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    for attempt in range(tries):
        try:
            r = HTTP.get(url, params=params, timeout=(8, 30))
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
            wait = min(int(r.headers.get("Retry-After", 5) or 5), 30)
            print(f"      rate-limited by {url.split('/')[2]}, waiting {wait}s", flush=True)
            time.sleep(wait); continue
        if r.status_code >= 500:
            time.sleep(3); continue
        return None
    print(f"      gave up on {url}", flush=True)
    return None


def jolpica(path, **params):
    time.sleep(0.3)                                # Jolpica allows ~4 requests a second
    d = _get(f"https://api.jolpi.ca/ergast/f1/{path}.json", {"limit": 100, **params})
    return d["MRData"] if d else {}


def openf1(endpoint, **params):
    time.sleep(0.3)
    return _get(f"https://api.openf1.org/v1/{endpoint}", params) or []


def secs(t):
    """'1:12.345' or '72.345' -> 72.345"""
    if not t or not isinstance(t, str):
        return np.nan
    p = t.split(":")
    try:
        return sum(float(x) * 60 ** i for i, x in enumerate(reversed(p)))
    except ValueError:
        return np.nan


# ---------------------------------------------------------------- fetching
def race_info(year):
    races = jolpica(f"{year}/circuits/interlagos/results").get("RaceTable", {}).get("Races", [])
    if not races:
        return None
    r = races[0]
    res = []
    for x in r["Results"]:
        res.append({"driver": x["Driver"]["driverId"], "code": x["Driver"].get("code", x["Driver"]["familyName"][:3].upper()),
                    "team": x["Constructor"]["name"], "grid": int(x["grid"]), "pos": int(x["position"]),
                    "status": x["status"], "laps": int(x["laps"]), "points": float(x["points"]),
                    "time": x.get("Time", {}).get("time"), "fl_rank": int(x.get("FastestLap", {}).get("rank", 0) or 0),
                    "fl_time": x.get("FastestLap", {}).get("Time", {}).get("time")})
    return {"year": year, "round": int(r["round"]), "date": r["date"], "time": r.get("time", "17:00:00Z"),
            "name": r["raceName"], "results": pd.DataFrame(res)}


def lap_times(year, rnd):
    """All drivers, all laps. Jolpica pages by timing rows, so keep going until a page is empty."""
    rows, offset = [], 0
    while True:
        md = jolpica(f"{year}/{rnd}/laps", offset=offset)
        races = md.get("RaceTable", {}).get("Races", []) if md else []
        laps = races[0].get("Laps", []) if races else []   # a page past the end comes back with no races
        n = 0
        for lap in laps:
            for t in lap["Timings"]:
                rows.append({"lap": int(lap["number"]), "driver": t["driverId"], "position": int(t["position"]),
                             "lap_s": secs(t["time"])})
                n += 1
        total = int(md.get("total", 0)) if md else 0
        offset += 100
        if n == 0 or offset >= total:
            break
    return pd.DataFrame(rows).drop_duplicates(["lap", "driver"])


def pit_stops(year, rnd):
    md = jolpica(f"{year}/{rnd}/pitstops")
    races = md.get("RaceTable", {}).get("Races", []) if md else []
    rows = [{"driver": p["driverId"], "lap": int(p["lap"]), "stop": int(p["stop"]),
             "pit_lane_s": secs(p.get("duration"))} for p in (races[0]["PitStops"] if races else [])]
    return pd.DataFrame(rows, columns=["driver", "lap", "stop", "pit_lane_s"])


def rain(date, hours):
    d = _get("https://archive-api.open-meteo.com/v1/archive",
             {"latitude": INTERLAGOS[0], "longitude": INTERLAGOS[1], "start_date": date, "end_date": date,
              "hourly": "precipitation,temperature_2m", "timezone": "America/Sao_Paulo"})
    if not d or "hourly" not in d:
        return np.nan, np.nan
    h = pd.DataFrame(d["hourly"])
    h = h[pd.to_datetime(h["time"]).dt.hour.isin(hours)]
    return round(h["precipitation"].sum(), 1), round(h["temperature_2m"].mean(), 1)


def openf1_extras(year):
    """Tyre stints, race control and OpenF1 overtakes for his race (2023 onwards)."""
    sess = openf1("sessions", year=year, country_name="Brazil", session_name="Race")
    if not sess:
        return [pd.DataFrame()] * 3
    sk = sess[-1]["session_key"]
    drv = [d for d in openf1("drivers", session_key=sk) if d.get("name_acronym") == CODE]
    if not drv:
        return [pd.DataFrame()] * 3
    num = drv[0]["driver_number"]
    who = {d["driver_number"]: d.get("name_acronym") for d in openf1("drivers", session_key=sk)}
    st = pd.DataFrame(openf1("stints", session_key=sk, driver_number=num))
    rc = pd.DataFrame(openf1("race_control", session_key=sk))
    if not rc.empty:
        msg = rc["message"].fillna("").str.upper()
        keep = (rc["category"].isin(["SafetyCar"]) | msg.str.contains("RED FLAG|VIRTUAL SAFETY|SAFETY CAR")
                | rc.get("driver_number", pd.Series(dtype=float)).eq(num) | msg.str.contains(f"CAR {num} "))
        rc = rc[keep][[c for c in ["date", "lap_number", "category", "flag", "message"] if c in rc.columns]]
    ov = pd.DataFrame(openf1("overtakes", session_key=sk))
    if not ov.empty and {"overtaking_driver_number", "overtaken_driver_number"} <= set(ov.columns):
        ov = ov[(ov["overtaking_driver_number"] == num) | (ov["overtaken_driver_number"] == num)].copy()
        ov["overtaking"] = ov["overtaking_driver_number"].map(who)
        ov["overtaken"] = ov["overtaken_driver_number"].map(who)
    else:
        ov = pd.DataFrame()
    for df in (st, rc, ov):
        if not df.empty:
            df.insert(0, "year", year)
    return st, rc, ov


# ---------------------------------------------------------------- analysis (pure pandas, testable offline)
def analyse(year, res, laps, pits):
    """Returns (summary row, his laps, his place changes, his stops) for one race."""
    me = res[res["driver"] == DRIVER_ID].iloc[0]
    code = dict(zip(res["driver"], res["code"]))
    laps = laps.sort_values(["driver", "lap"]).copy()
    laps["cum_s"] = laps.groupby("driver")["lap_s"].cumsum()
    pitted = set(zip(pits["driver"], pits["lap"]))
    last_lap = laps.groupby("driver")["lap"].max().to_dict()

    # gaps: to leader, to car ahead, to car behind, at the end of every lap
    gaps = []
    for lap, g in laps.groupby("lap"):
        g = g.sort_values("position")
        mine = g[g["driver"] == DRIVER_ID]
        if mine.empty:
            continue
        m = mine.iloc[0]
        lead = g["cum_s"].min()
        ahead = g[g["position"] == m["position"] - 1]["cum_s"]
        behind = g[g["position"] == m["position"] + 1]["cum_s"]
        gaps.append({"lap": lap, "gap_leader_s": round(m["cum_s"] - lead, 3),
                     "gap_ahead_s": round(m["cum_s"] - ahead.iloc[0], 3) if len(ahead) else np.nan,
                     "gap_behind_s": round(behind.iloc[0] - m["cum_s"], 3) if len(behind) else np.nan})
    my = laps[laps["driver"] == DRIVER_ID][["lap", "position", "lap_s"]].merge(pd.DataFrame(gaps), on="lap", how="left")
    my["pit_lap"] = my["lap"].map(lambda l: (DRIVER_ID, l) in pitted)
    my.insert(0, "year", year)

    # place changes lap by lap, against named cars
    order = {lap: g.sort_values("position")["driver"].tolist() for lap, g in laps.groupby("lap")}
    grid = res[res["grid"] > 0].sort_values("grid")["driver"].tolist()
    if me["grid"] == 0:                            # pit-lane start: behind everyone
        grid = grid + [DRIVER_ID]
    order[0] = grid
    changes = []
    for lap in sorted(k for k in order if k > 0):
        prev, now = order.get(lap - 1), order[lap]
        if not prev or DRIVER_ID not in now or DRIVER_ID not in prev:
            continue
        p_ahead = set(prev[:prev.index(DRIVER_ID)])
        n_ahead = set(now[:now.index(DRIVER_ID)])
        for d in p_ahead - n_ahead:                # he moved ahead of d
            if lap == 1:
                why = "start / lap 1"
            elif last_lap.get(d, 0) < lap:
                why = "retirement"
            elif (d, lap) in pitted or (d, lap - 1) in pitted:
                why = "pit cycle"
            else:
                why = "on track"
            changes.append({"year": year, "lap": lap, "change": "gained", "other": code.get(d, d), "cause": why})
        for d in n_ahead - p_ahead:                # d moved ahead of him
            if lap == 1:
                why = "start / lap 1"
            elif (DRIVER_ID, lap) in pitted or (DRIVER_ID, lap - 1) in pitted:
                why = "own stop"
            elif (d, lap) in pitted or (d, lap - 1) in pitted:
                why = "pit cycle"
            else:
                why = "on track"
            changes.append({"year": year, "lap": lap, "change": "lost", "other": code.get(d, d), "cause": why})
    ch = pd.DataFrame(changes, columns=["year", "lap", "change", "other", "cause"])

    # his stops with positions either side
    ps = pits[pits["driver"] == DRIVER_ID].copy()
    pos_at = dict(zip(my["lap"], my["position"]))
    ps["pos_before"] = ps["lap"].map(lambda l: pos_at.get(l - 1))
    ps["pos_after"] = ps["lap"].map(lambda l: pos_at.get(l + 1))
    ps.insert(0, "year", year)
    ps = ps.drop(columns=["driver"])

    winner = res.sort_values("pos").iloc[0]
    on_track = ch[(ch["cause"] == "on track") & (ch["change"] == "gained")]
    summary = {
        "year": year, "team": me["team"], "grid": int(me["grid"]) if me["grid"] > 0 else "pit lane",
        "finish": int(me["pos"]), "status": me["status"], "points": me["points"],
        "places_gained": (int(me["grid"]) if me["grid"] > 0 else len(res)) - int(me["pos"]),
        "laps_led": int((my["position"] == 1).sum()), "laps": int(me["laps"]),
        "best_position": int(my["position"].min()), "worst_position": int(my["position"].max()),
        "best_lap": me["fl_time"], "best_lap_rank": me["fl_rank"],
        "stops": len(ps), "on_track_passes": len(on_track),
        "passed_on_track": " ".join(on_track["other"]),
        "lost_on_track": len(ch[(ch["cause"] == "on track") & (ch["change"] == "lost")]),
        "winner": winner["code"], "winner_team": winner["team"],
        "gap_to_winner": "" if me["pos"] == 1 else (me["time"] or me["status"]),
        "win_margin": res[res["pos"] == 2]["time"].iloc[0] if me["pos"] == 1 and (res["pos"] == 2).any() else "",
    }
    return summary, my, ch, ps


def main():
    global USE_CACHE
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs=2, type=int, default=[2015, 2025])
    ap.add_argument("--cache", action="store_true")
    a = ap.parse_args()
    USE_CACHE = a.cache
    os.makedirs(OUT, exist_ok=True)

    summ, all_laps, all_ch, all_ps, all_pos, sts, rcs, ovs = [], [], [], [], [], [], [], []
    for y in range(a.years[0], a.years[1] + 1):
        with step(f"{y}"):
            info = race_info(y)
            if not info:
                print("      no race at Interlagos"); continue
            res = info["results"]
            if DRIVER_ID not in set(res["driver"]):
                print("      Verstappen did not race"); continue
            laps = lap_times(y, info["round"])
            pits = pit_stops(y, info["round"])
            if laps.empty:
                print("      no lap data"); continue
            s, my, ch, ps = analyse(y, res, laps, pits)
            # UTC -> São Paulo local hour; Brazil used summer time (UTC-2) until 2019, so let pandas do it
            start_h = pd.Timestamp(f"{info['date']}T{info['time']}").tz_convert("America/Sao_Paulo").hour
            s["rain_mm"], s["air_c"] = rain(info["date"], [start_h, (start_h + 1) % 24])
            if y >= 2023:
                st, rc, ov = openf1_extras(y)
                if not st.empty and "compound" in st.columns:
                    s["compounds"] = "-".join(st.sort_values("stint_number")["compound"].str[0])
                sts.append(st); rcs.append(rc); ovs.append(ov)
            summ.append(s); all_laps.append(my); all_ch.append(ch); all_ps.append(ps)
            p = laps[["lap", "driver", "position"]].copy()
            p["code"] = p["driver"].map(dict(zip(res["driver"], res["code"])))
            p.insert(0, "year", y)
            all_pos.append(p.drop(columns=["driver"]))
            print(f"      grid {s['grid']} -> P{s['finish']}, {s['on_track_passes']} on-track passes, "
                  f"{s['laps_led']} laps led", flush=True)

    def save(frames, name):
        frames = [f for f in frames if f is not None and not f.empty]
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(os.path.join(OUT, name), index=False)
            print(f"  wrote {name}")

    save([pd.DataFrame(summ)], "ver_brazil_summary.csv")
    save(all_laps, "ver_brazil_laps.csv")
    save(all_ch, "ver_brazil_passes.csv")
    save(all_ps, "ver_brazil_pitstops.csv")
    save(sts, "ver_brazil_stints.csv")
    save(rcs, "ver_brazil_race_control.csv")
    save(ovs, "ver_brazil_overtakes_openf1.csv")
    save(all_pos, "all_drivers_positions.csv")
    print(f"\nCSVs in {OUT}")


if __name__ == "__main__":
    main()
