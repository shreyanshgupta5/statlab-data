#!/usr/bin/env python3
"""Every driver's lap times, 2024 São Paulo GP (Jolpica) -> data/brazil_2024_all_laps.csv"""
import time, requests, pandas as pd
API = "https://api.jolpi.ca/ergast/f1/2024/21"
S = requests.Session(); S.headers["User-Agent"] = "statlab/1.0"

def get(path, **p):
    for i in range(4):
        r = S.get(f"{API}/{path}.json", params={"limit": 100, **p}, timeout=30)
        if r.status_code == 200:
            time.sleep(0.3); return r.json()["MRData"]
        time.sleep(3 * (i + 1))
    raise SystemExit(f"failed {path}")

def secs(t):
    m, s = (t.split(":") + [None])[:2] if ":" in t else (0, t)
    return float(m) * 60 + float(s) if ":" in t else float(t)

res = get("results")["RaceTable"]["Races"][0]
assert "Paulo" in res["raceName"] or "Brazil" in res["raceName"], res["raceName"]
code = {x["Driver"]["driverId"]: x["Driver"]["code"] for x in res["Results"]}
rows, off = [], 0
while True:
    md = get("laps", offset=off)
    laps = md["RaceTable"]["Races"][0]["Laps"] if md["RaceTable"]["Races"] else []
    n = 0
    for lap in laps:
        for t in lap["Timings"]:
            rows.append({"lap": int(lap["number"]), "driver": code.get(t["driverId"], t["driverId"]),
                         "position": int(t["position"]), "lap_s": secs(t["time"])}); n += 1
    off += 100
    if n == 0 or off >= int(md["total"]):
        break
df = pd.DataFrame(rows).drop_duplicates(["lap", "driver"])
pits = get("pitstops")["RaceTable"]["Races"][0].get("PitStops", [])
pin = {(code.get(p["driverId"]), int(p["lap"])) for p in pits}
df["pit_in"] = [(d, l) in pin for d, l in zip(df.driver, df.lap)]
df["pit_out"] = [(d, l - 1) in pin for d, l in zip(df.driver, df.lap)]
import os; os.makedirs("data", exist_ok=True)
df.sort_values(["lap", "position"]).to_csv("data/brazil_2024_all_laps.csv", index=False)
print(len(df), "laps,", df.driver.nunique(), "drivers,", len(pits), "pit stops")
