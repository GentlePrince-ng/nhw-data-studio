"""Build a 10,000-row test workbook that looks like a real Kobo health export.

Every defect here is one that has actually turned up in an NHW dataset, so the
sample doubles as the regression test: duplicated submissions, a state name in
three casings, 99-as-don't-know, a column somebody stopped filling in halfway
through, a select_multiple exported as one column per option, and a fortnight
where nobody collected anything.

    python samples/make_sample.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import geo  # noqa: E402  — for real LGA names, see below

N = 10_000
rng = np.random.default_rng(20260821)
OUT = Path(__file__).resolve().parent / "sample_phc_survey.xlsx"

STATES = ["Kano", "Lagos", "Bauchi", "Kebbi", "Rivers", "Enugu", "Borno",
          "Sokoto", "Oyo", "FCT", "Cross River", "Ebonyi"]
WEIGHTS = np.array([22, 16, 11, 9, 8, 7, 6, 6, 5, 4, 3, 3], dtype=float)
WEIGHTS /= WEIGHTS.sum()

state = rng.choice(STATES, N, p=WEIGHTS)
# One state arrives in three casings — this splits Kano into three categories in
# any pivot table until somebody notices.
state = np.array([s if s != "Kano" or i % 7 else ("kano " if i % 14 else "KANO")
                  for i, s in enumerate(state)])

# Real LGA names, drawn from the vendored boundary file so the LGA map and the
# state-qualified join are exercised by the sample rather than only by hope. Two
# of the states below share an LGA name with another state, which is the case
# that makes an unqualified LGA join wrong.
_lgas = {}
for f in geo._load("lga").get("features", []):
    _lgas.setdefault(geo.canon_state(f["properties"]["state"]), []).append(
        f["properties"]["lga"])
if _lgas:
    lga_names = []
    for s_val in state:
        pool = _lgas.get(geo.canon_state(s_val)) or ["Unknown"]
        lga_names.append(pool[int(rng.integers(0, len(pool)))])
else:
    # No boundary files built (they are not in the repository). Invented LGA
    # names keep the rest of the sample honest — every other planted defect is
    # unaffected, and the LGA map would not draw either way.
    print("No boundary files found: using placeholder LGA names. Build the "
          "boundaries (static/build_boundaries.py) for real ones.")
    lga_names = [f"LGA {int(rng.integers(1, 18))}" for _ in range(len(state))]

# Collection runs Feb–Jul 2026 with a dead fortnight in May.
days = rng.integers(0, 150, N)
start = pd.Timestamp("2026-02-02")
dates = start + pd.to_timedelta(days, unit="D")
gap = (dates >= "2026-05-04") & (dates < "2026-05-19")
dates = pd.Series(dates).where(~gap, dates + pd.Timedelta(days=15))

sex = rng.choice(["Female", "Male"], N, p=[0.63, 0.37])
age = rng.normal(34, 12, N).clip(15, 88).round()
# Don't-know coded 99, as every survey firm in Abuja does it.
age = np.where(rng.random(N) < 0.06, 99, age)

facility_type = rng.choice(["PHC centre", "Health post", "Comprehensive centre",
                            "Private clinic"], N, p=[0.52, 0.26, 0.13, 0.09])
# Wait time genuinely differs by facility type — the tool should find this.
base_wait = {"PHC centre": 48, "Health post": 71, "Comprehensive centre": 34,
             "Private clinic": 19}
wait = np.array([rng.normal(base_wait[f], 16) for f in facility_type]).clip(2, 240)
# Staff count correlates with wait time, inversely.
staff = (14 - wait / 12 + rng.normal(0, 1.6, N)).clip(1, 22).round()

visits = rng.poisson(2.6, N)
satisfied = np.where(wait < 45,
                     rng.choice(["Yes", "No"], N, p=[0.78, 0.22]),
                     rng.choice(["Yes", "No"], N, p=[0.41, 0.59]))

# A column the field team stopped filling from about the halfway mark.
phone = np.where(np.arange(N) < N * 0.42,
                 [f"080{rng.integers(10000000, 99999999)}" for _ in range(N)],
                 None)

df = pd.DataFrame({
    "_id": np.arange(100001, 100001 + N),
    "submission_date": dates.dt.strftime("%Y-%m-%d %H:%M:%S"),
    "state": state,
    "lga": lga_names,
    "facility_type": facility_type,
    "respondent_sex": sex,
    "respondent_age": age.astype(int),
    "visits_last_year": visits,
    "wait_minutes": wait.round(1),
    "staff_on_duty": staff.astype(int),
    "satisfied_with_visit": satisfied,
    "callback_number": phone,
    "enumerator_notes": [
        rng.choice(["Facility busy at time of visit.",
                    "Respondent declined some questions.",
                    "Interview completed without issue.",
                    "Drug shelf was visibly empty; noted for follow up.", ""])
        for _ in range(N)],
    "form_version": "v3",                       # constant: should be dropped
    "audit_url": None,                          # empty: should be dropped
})

# Kobo select_multiple: one binary column per option, named parent/option.
barriers = ["cost", "distance", "no_drugs", "staff_attitude", "long_wait",
            "no_transport"]
probs = [0.44, 0.31, 0.38, 0.19, 0.35, 0.22]
for b, p in zip(barriers, probs):
    df[f"barriers_to_care/{b}"] = (rng.random(N) < p).astype(int)

# 180 duplicated submissions, the way a re-synced tablet produces them.
dupes = df.sample(180, random_state=3)
df = pd.concat([df, dupes], ignore_index=True)

with pd.ExcelWriter(OUT, engine="xlsxwriter") as xw:
    df.to_excel(xw, sheet_name="survey_data", index=False)
    pd.DataFrame({
        "note": ["Second sheet, so the sheet picker gets exercised."],
    }).to_excel(xw, sheet_name="readme", index=False)

print(f"wrote {OUT}  ({len(df):,} rows x {len(df.columns)} cols)")
