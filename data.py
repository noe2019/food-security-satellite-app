"""Simulated monthly NDVI and rainfall for KwaZulu-Natal districts (2015–2022)."""
import numpy as np
import pandas as pd

DISTRICTS = {"eThekwini": (-29.86, 30.98), "Ugu": (-30.6, 30.3), "uMgungundlovu": (-29.6, 30.3), "uThukela": (-28.6, 29.8),
             "uMzinyathi": (-28.4, 30.6), "Amajuba": (-27.8, 30.0), "Zululand": (-27.8, 31.4), "uMkhanyakude": (-27.6, 32.3),
             "King Cetshwayo": (-28.6, 31.6), "iLembe": (-29.3, 31.2), "Harry Gwala": (-30.0, 29.6)}
SYSTEMS = ["Smallholder", "Commercial"]
LOCKDOWN = ("2020-03-27", "2020-06-30")


def simulate(seed=41):
    r = np.random.default_rng(seed)
    months = pd.date_range("2015-01-01", "2022-12-01", freq="MS")
    # seasonal rainfall (summer rainfall region) with 2015-16 El Niño drought and 2019 dry spell
    clim_rain = 60 + 70 * np.cos((months.month.to_numpy() - 1) / 12 * 2 * np.pi)
    drought = np.where((months >= "2015-06-01") & (months <= "2016-06-01"), .45,
                       np.where((months >= "2019-01-01") & (months <= "2019-10-01"), .7, 1.0))
    lock = (months >= pd.Timestamp(LOCKDOWN[0]).replace(day=1)) & (months <= LOCKDOWN[1])
    rows = []
    for d, (lat, lon) in DISTRICTS.items():
        wet = 1 + (lon - 30.5) * .08 + r.normal(0, .05)
        rain = (clim_rain * drought * wet * r.lognormal(0, .25, len(months))).clip(2)
        for s in SYSTEMS:
            base = .48 if s == "Commercial" else .40
            irrig = .6 if s == "Commercial" else .15  # commercial farms buffer drought with irrigation
            ndvi = (base + .17 * np.cos((months.month.to_numpy() - 2) / 12 * 2 * np.pi)
                    - (1 - drought) * (1 - irrig) * .25
                    - np.where(lock, .045 if s == "Smallholder" else .012, 0)
                    + r.normal(0, .015, len(months))).clip(.05, .9)
            rows.append(pd.DataFrame({"date": months, "district": d, "system": s, "lat": lat, "lon": lon,
                                      "rainfall_mm": rain.round(1), "ndvi": ndvi.round(4)}))
    df = pd.concat(rows, ignore_index=True)
    df["lockdown"] = df.date.between(pd.Timestamp(LOCKDOWN[0]).replace(day=1), LOCKDOWN[1]).astype(int)
    return df
