"""NEXUS DELTA demo app  ·  run:  streamlit run app.py   (needs out/results.json, out/taxonomy.json, out/jd_scanner_m2.joblib from notebooks 01-06)
Tabs: Evidence Matrix (org lens) · JD Scanner (T2) · Student lens (PROPOSED rule-based).  No remote API calls; fully offline."""
import json, re
import numpy as np, pandas as pd, streamlit as st, joblib
st.set_page_config(page_title="NEXUS DELTA", layout="wide")
R = json.load(open("out/results.json")); TAX = json.load(open("out/taxonomy.json"))["TAX"]
M = pd.DataFrame(R["matrix"]); BANDS = ["0-3","3-6","6-10","10-15","15-25","25-50"]
st.title("NEXUS DELTA — Evidence-First Workforce Capability Decision Engine")
st.caption("Team DSA · Build for Bharat 2.0 · every number is an association from the organiser datasets, not a causal effect.")
t1, t2, t3 = st.tabs(["Evidence Matrix", "JD Scanner (T2)", "Student lens (proposed)"])
with t1:
    st.dataframe(M[["capability","delta","internal","OR_primary","OR_wide","market","headroom","Ek","Ek_lo","Ek_hi","verdict"]].round(2), width="stretch", hide_index=True)
    d = R["decision"]; st.info(f"Lead lever: **{d['lead']}** — beats {d['next']} in {d['p_beat']:.0%} of 1,000 bootstrap refits ({'tie' if d['tie'] else 'clear winner'}).")
    st.image("figs/fig06_decision.png")
with t2:
    model = joblib.load("out/jd_scanner_m2.joblib")
    c1, c2 = st.columns(2)
    title = c1.text_input("Designation", "Senior Data Scientist"); skills = c1.text_area("Key skills / JD text", "python, machine learning, nlp, sql")
    lo, hi = c2.slider("Experience (yrs)", 0, 20, (4, 8)); sen = c2.selectbox("Seniority", ["junior","mid","senior","lead_mgr","head"], 2)
    loc = c2.selectbox("Location tier", ["Tier1","Tier2_3","International"]); rf = c2.selectbox("Role family", ["DS_ML","DE","BA","DA_BI","other"])
    t = (title + " " + skills).lower(); tags = {k: int(bool(re.search(p, t))) for k, p in TAX.items()}
    row = dict(job_desig=title, text=t, exp_min=lo, exp_max=hi, exp_span=hi-lo, seniority=sen, loc_tier=loc, role_family=rf, jobtype_missing=1, trunc_skills=0, desc_missing=1, **{f"{k}_w": v for k, v in tags.items()})
    pr = model.predict_proba(pd.DataFrame([row]))[0]
    st.write("Capability tags:", ", ".join(k for k, v in tags.items() if v) or "none")
    st.bar_chart(pd.Series(pr, index=BANDS, name="P(salary band, LPA)"))
    st.caption("Output is a probability distribution (exact-band accuracy ≈ 44%, within ±1 band ≈ 85%) — triage, not a salary oracle.")
with t3:
    st.caption("PROPOSED extension: uses market evidence + headroom from your own entered scores. It never uses the single-company JDS outcome model to predict an individual.")
    names = list(M.capability); scores = {n: st.slider(n, 1.0, 5.0, 3.5, 0.1) for n in names}
    rows = []
    for r in M.itertuples():
        gap = 5 - scores[r.capability]; w = np.log(r.OR_primary)
        rows.append(dict(capability=r.capability, your_score=scores[r.capability], gap=round(gap, 1), market_OR=round(r.OR_primary, 2),
                         priority=round(gap * max(w, 0), 3), note="reframe: invest in visual analytics / decision communication, not generic MIS" if r.verdict == "FUND WITH REFRAME" else ("weak evidence" if r.verdict == "DEPRIORITISE" else "")))
    st.dataframe(pd.DataFrame(rows).sort_values("priority", ascending=False), hide_index=True, width="stretch")
