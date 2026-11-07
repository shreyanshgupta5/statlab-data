#!/usr/bin/env python3
"""2016 Brazilian GP, everything Jolpica + Open-Meteo have -> data/brazil2016/*.csv"""
import os, time, requests, pandas as pd
API = "https://api.jolpi.ca/ergast/f1"; OUT = "data/brazil2016"; os.makedirs(OUT, exist_ok=True)
S = requests.Session(); S.headers["User-Agent"] = "statlab/1.0"
def get(path, **p):
    for i in range(5):
        r = S.get(f"{API}/{path}.json", params={"limit": 100, **p}, timeout=30)
        if r.status_code == 200:
            time.sleep(0.35); return r.json()["MRData"]
        time.sleep(4 * (i + 1))
    raise SystemExit(f"failed {path}")
def secs(t):
    if not t: return None
    p = t.split(":"); return sum(float(x) * 60 ** i for i, x in enumerate(reversed(p)))
race = get("2016/circuits/interlagos/results")["RaceTable"]["Races"][0]; rnd = int(race["round"])
print(race["raceName"], race["date"], "round", rnd)
code = {x["Driver"]["driverId"]: x["Driver"]["code"] for x in race["Results"]}
pd.DataFrame([{"pos": int(x["position"]), "code": x["Driver"]["code"], "driver": x["Driver"]["givenName"] + " " + x["Driver"]["familyName"],
  "team": x["Constructor"]["name"], "grid": int(x["grid"]), "laps": int(x["laps"]), "status": x["status"], "points": float(x["points"]),
  "time": x.get("Time", {}).get("time"), "fl_rank": x.get("FastestLap", {}).get("rank"), "fl_lap": x.get("FastestLap", {}).get("lap"),
  "fl_time": x.get("FastestLap", {}).get("Time", {}).get("time")} for x in race["Results"]]).to_csv(f"{OUT}/results.csv", index=False)
q = get(f"2016/{rnd}/qualifying")["RaceTable"]["Races"][0]["QualifyingResults"]
pd.DataFrame([{"pos": int(x["position"]), "code": x["Driver"]["code"], "team": x["Constructor"]["name"],
  "q1": x.get("Q1"), "q2": x.get("Q2"), "q3": x.get("Q3")} for x in q]).to_csv(f"{OUT}/qualifying.csv", index=False)
rows, off = [], 0
while True:
    md = get(f"2016/{rnd}/laps", offset=off); laps = md["RaceTable"]["Races"][0]["Laps"] if md["RaceTable"]["Races"] else []
    n = 0
    for lap in laps:
        for t in lap["Timings"]:
            rows.append({"lap": int(lap["number"]), "code": code.get(t["driverId"], t["driverId"]), "position": int(t["position"]), "lap_s": secs(t["time"])}); n += 1
    off += 100
    if n == 0 or off >= int(md["total"]): break
laps = pd.DataFrame(rows).drop_duplicates(["lap", "code"])
pits = get(f"2016/{rnd}/pitstops")["RaceTable"]["Races"][0].get("PitStops", [])
pits = pd.DataFrame([{"code": code.get(p["driverId"]), "lap": int(p["lap"]), "stop": int(p["stop"]), "time_of_day": p["time"],
  "duration_s": secs(p["duration"]) if ":" not in p["duration"] or True else None, "duration_raw": p["duration"]} for p in pits])
pits.to_csv(f"{OUT}/pitstops.csv", index=False)
pin = set(zip(pits.code, pits.lap)); laps["pit_in"] = [(c, l) in pin for c, l in zip(laps.code, laps.lap)]
laps["pit_out"] = [(c, l - 1) in pin for c, l in zip(laps.code, laps.lap)]
laps.sort_values(["lap", "position"]).to_csv(f"{OUT}/laps.csv", index=False)
for label, r in [("before", rnd - 1), ("after", rnd)]:
    st = get(f"2016/{r}/driverStandings")["StandingsTable"]["StandingsLists"][0]["DriverStandings"]
    pd.DataFrame([{"pos": int(s["position"]), "code": s["Driver"]["code"], "points": float(s["points"]), "wins": int(s["wins"])} for s in st[:10]]).to_csv(f"{OUT}/standings_{label}.csv", index=False)
w = requests.get("https://archive-api.open-meteo.com/v1/archive", params={"latitude": -23.7036, "longitude": -46.6997,
  "start_date": race["date"], "end_date": race["date"], "hourly": "temperature_2m,precipitation,relative_humidity_2m,wind_speed_10m",
  "timezone": "America/Sao_Paulo"}, timeout=30).json()
pd.DataFrame(w["hourly"]).to_csv(f"{OUT}/weather_hourly.csv", index=False)
print(len(laps), "laps,", laps.code.nunique(), "drivers,", len(pits), "pit entries")
