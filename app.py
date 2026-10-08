"""AI Disaster Alert & Emergency Assistant - Streamlit app (academic prototype)."""
import os, json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.preprocessing import (load_all_tables, pick_tables, detect_schema, detect_flood_target, describe_dataset,
                               make_demo_datasets, FRIENDLY, DATA_DIR)
from src.flood_prediction import train_flood_model, predict_flood, explain_flood
from src.earthquake_risk import QuakeIndicator, DISCLAIMER as QUAKE_DISCLAIMER
from src.alert_engine import (overall_alert, banner_html, STYLE, FLOOD_ACTIONS, QUAKE_ACTIONS,
                              flood_alert_level, quake_alert_level)
from src.assistant import answer, llm_available

st.set_page_config(page_title="AI Disaster Alert & Emergency Assistant", page_icon="🌍", layout="wide")

DISCLAIMER = ("This application is an academic AI decision-support prototype. It does not guarantee prediction of natural "
              "disasters. Earthquake risk indicators are probabilistic and should not be interpreted as exact earthquake "
              "predictions. Flood risk estimates depend on the quality and coverage of the available data. Always follow "
              "official government and emergency-management warnings and evacuation instructions.")
CONTACT_FILE = os.path.join(DATA_DIR, "emergency_contacts.json")
LEVEL_COLORS = {"LOW": "#16a34a", "MODERATE": "#ca8a04", "HIGH": "#ea580c", "EXTREME": "#dc2626"}

st.markdown("""<style>
.block-container{padding-top:1.5rem}
.card{background:linear-gradient(135deg,#0f172a,#1e3a8a);color:white;padding:18px;border-radius:14px;text-align:center}
.card h4{margin:0;font-size:.9rem;opacity:.8;font-weight:500}.card h2{margin:6px 0 0 0;font-size:1.8rem}
.demo{background:#7c3aed22;border:1px dashed #7c3aed;padding:8px 12px;border-radius:8px;font-size:.9rem}
.disc{background:#fef3c7;border-left:6px solid #f59e0b;padding:10px 14px;border-radius:8px;color:#78350f;font-size:.85rem}
</style>""", unsafe_allow_html=True)


# ------------------------------------------------------------------ data & models
@st.cache_data(show_spinner=False)
def get_data(version):
    tables = load_all_tables()
    fname, qname = pick_tables(tables)
    info = {"flood_source": None, "quake_source": None, "synthetic_flood": False, "synthetic_quake": False,
            "files": {k: v.shape for k, v in tables.items()}}
    demo_f, demo_q = make_demo_datasets()
    if fname:
        flood_df, info["flood_source"] = tables[fname], fname
    else:
        flood_df, info["flood_source"], info["synthetic_flood"] = demo_f, "SYNTHETIC demo dataset", True
    if qname:
        quake_df, info["quake_source"] = tables[qname], qname
    else:
        quake_df, info["quake_source"], info["synthetic_quake"] = demo_q, "SYNTHETIC demo catalogue", True
    return flood_df, quake_df, info


@st.cache_resource(show_spinner="Training & comparing ML models ...")
def get_flood_model(version):
    flood_df, _, _ = get_data(version)
    return train_flood_model(flood_df)


@st.cache_resource(show_spinner=False)
def get_quake(version):
    _, quake_df, _ = get_data(version)
    try:
        return QuakeIndicator(quake_df), None
    except Exception as e:
        return None, str(e)


if "ver" not in st.session_state:
    st.session_state.ver = 0
ver = st.session_state.ver
flood_df, quake_df, info = get_data(ver)
try:
    FM = get_flood_model(ver)
    fm_err = None
except Exception as e:
    FM, fm_err = None, str(e)
QI, qi_err = get_quake(ver)
fschema = detect_schema(flood_df)

# ------------------------------------------------------------------ sidebar
st.sidebar.title("🌍 Disaster AI")
page = st.sidebar.radio("Navigate", ["🏠 Dashboard", "🌊 Flood Prediction", "🌎 Earthquake Risk",
                                     "🤖 Disaster AI Assistant", "🚨 Emergency Guide", "📊 Data Analysis",
                                     "🗺️ Risk Map", "ℹ️ About Project"])
st.sidebar.markdown("---")
up = st.sidebar.file_uploader("Upload your dataset (.zip / .csv / .xlsx)", type=["zip", "csv", "xlsx"])
if up is not None and st.sidebar.button("Use this dataset"):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, up.name), "wb") as f:
        f.write(up.getbuffer())
    st.session_state.ver += 1
    st.cache_data.clear(); st.cache_resource.clear()
    st.rerun()
if info["synthetic_flood"]:
    st.sidebar.warning("No flood dataset found in data/. Using SYNTHETIC demo data (for demo only).")
else:
    st.sidebar.success(f"Flood data: {info['flood_source']}")
st.sidebar.caption(("Quake data: " + info["quake_source"]) if QI else "No earthquake catalogue available")
st.sidebar.markdown("**Mode:** " + ("🟣 DEMO (synthetic)" if info["synthetic_flood"] else "📁 Your dataset"))
st.sidebar.markdown(f"<div class='disc'>{DISCLAIMER}</div>", unsafe_allow_html=True)

if info["synthetic_flood"] or info["synthetic_quake"]:
    st.markdown("<div class='demo'>🟣 <b>DEMO MODE</b> - parts of this view use SYNTHETIC data for demonstration. "
                "It is NOT real-time or official disaster prediction.</div>", unsafe_allow_html=True)
if fm_err:
    st.error(f"Flood model could not be trained: {fm_err}")


# ------------------------------------------------------------------ helpers
def input_state_key(f):
    return f"in_{f}"


def feature_inputs(prefix=""):
    """Render sliders/number inputs for model features; return dict of values."""
    vals = {}
    feats = FM["features"]
    cols = st.columns(3)
    role_of = {v: k for k, v in fschema.items()}
    for i, f in enumerate(feats[:15]):
        lo, hi = float(FM["stats"]["min"][f]), float(FM["stats"]["max"][f])
        if lo == hi:
            hi = lo + 1
        key = prefix + input_state_key(f)
        if key not in st.session_state:
            st.session_state[key] = float(np.clip(FM["stats"]["median"][f], lo, hi))
        label = FRIENDLY.get(role_of.get(f), f) + (f"  ({f})" if role_of.get(f) and FRIENDLY.get(role_of.get(f)) != f else "")
        vals[f] = cols[i % 3].slider(label, lo, hi, key=key)
    return vals


def set_preset(prefix, q_by_role, default_q=0.5):
    role_of = {v: k for k, v in fschema.items()}
    for f in FM["features"][:15]:
        q = q_by_role.get(role_of.get(f), default_q)
        qs = FM["stats"]["q"]
        lo, hi = FM["stats"]["min"][f], FM["stats"]["max"][f]
        val = qs[q][f] if q in qs else FM["stats"]["median"][f]
        if q_by_role.get("_max", {}).get(role_of.get(f)):
            val = hi
        st.session_state[prefix + input_state_key(f)] = float(np.clip(val, lo, max(hi, lo + 1)))


def heavy_rain():
    set_preset("m_", {"rainfall": 0.97, "water_level": 0.97, "humidity": 0.9, "soil_moisture": 0.9, "pressure": 0.3, "wind": 0.9})


def normal_cond():
    set_preset("m_", {"rainfall": 0.3, "water_level": 0.3, "humidity": 0.3, "soil_moisture": 0.3})
    # also pick the dataset location with the fewest historical earthquakes nearby (calm scenario)
    if QI is not None and "lat" in fschema and "lon" in fschema and fschema["lat"] in FM["features"] and fschema["lon"] in FM["features"]:
        pts = flood_df[[fschema["lat"], fschema["lon"]]].dropna().sample(min(len(flood_df), 120), random_state=3)
        best = min(pts.itertuples(index=False), key=lambda r: QI._count(r[0], r[1]))
        st.session_state["m_" + input_state_key(fschema["lat"])] = float(best[0])
        st.session_state["m_" + input_state_key(fschema["lon"])] = float(best[1])


def gauge(p, title, color):
    fig = go.Figure(go.Indicator(mode="gauge+number", value=p * 100, number={"suffix": "%"},
                                 title={"text": title},
                                 gauge={"axis": {"range": [0, 100]}, "bar": {"color": color},
                                        "steps": [{"range": [0, 25], "color": "#dcfce7"}, {"range": [25, 50], "color": "#fef9c3"},
                                                  {"range": [50, 75], "color": "#fed7aa"}, {"range": [75, 100], "color": "#fecaca"}]}))
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def card(title, value):
    return f"<div class='card'><h4>{title}</h4><h2>{value}</h2></div>"


def show_why(expl, level):
    st.subheader(f"🔍 Why is Flood Risk {level}?")
    top = expl.head(5)
    for i, r in enumerate(top.itertuples(), 1):
        arrow = "⬆️" if r[4] == "increases" else "⬇️" if r[4] == "decreases" else "➖"
        st.markdown(f"**{i}. {r.Feature}** → {r.Impact} impact {arrow} (your value {r[2]} vs typical {r[3]})")
    st.caption("Explanation method: for each feature we replace your value with the dataset median and measure how much the "
               "model's risk (log-odds) changes. Bigger change = bigger influence. Global importance chart is on the Dashboard.")


def load_contacts():
    default = {"Local Emergency Services": "", "Police": "", "Ambulance": "", "Fire & Rescue": "", "Disaster Management Authority": ""}
    try:
        with open(CONTACT_FILE) as f:
            default.update(json.load(f))
    except Exception:
        pass
    return default


# ================================================================== DASHBOARD
if page == "🏠 Dashboard":
    st.title("🌍 AI Disaster Alert & Emergency Assistant")
    st.caption("AI-Based Earthquake & Flood Risk Prediction and Disaster Safety System")
    if FM is None:
        st.stop()
    role_of = {v: k for k, v in fschema.items()}
    latest = flood_df.iloc[-1]
    vals = {f: (latest[f] if pd.notna(latest[f]) else FM["stats"]["median"][f]) for f in FM["features"]}
    p, lvl = predict_flood(FM, vals)
    qa = None
    if QI and "lat" in fschema and "lon" in fschema and pd.notna(latest.get(fschema["lat"])):
        qa = QI.assess(float(latest[fschema["lat"]]), float(latest[fschema["lon"]]))
    ov = overall_alert(p, qa["score"] if qa else None)
    st.markdown(banner_html(ov["level"], f"OVERALL ALERT: {ov['level']}"), unsafe_allow_html=True)
    st.caption("Shown for the LATEST RECORD in the dataset (not live data).")
    c = st.columns(4)
    c[0].markdown(card("🌊 Flood Risk", f"{lvl} · {p*100:.0f}%"), unsafe_allow_html=True)
    c[1].markdown(card("🌎 Earthquake Indicator", f"{qa['level']} · {qa['score']*100:.0f}%" if qa else "N/A"), unsafe_allow_html=True)
    c[2].markdown(card("⚠️ Overall Alert", f"{STYLE[ov['level']][0]} {ov['level']}"), unsafe_allow_html=True)
    c[3].markdown(card("📊 Model ROC-AUC", f"{FM['metrics'].iloc[0]['ROC-AUC']:.3f}"), unsafe_allow_html=True)
    st.markdown("")
    dcol = fschema.get("date")
    xs = pd.to_datetime(flood_df[dcol], errors="coerce") if dcol else pd.Series(range(len(flood_df)))
    tabs = st.tabs(["Trends", "Risk distribution", "Historical events", "Model & features"])
    with tabs[0]:
        cc = st.columns(3)
        for col, role, name in zip(cc, ["rainfall", "temperature", "humidity"], ["Rainfall", "Temperature", "Humidity"]):
            if role in fschema:
                s = pd.DataFrame({"x": xs, "y": pd.to_numeric(flood_df[fschema[role]], errors="coerce")}).dropna()
                s = s.tail(1000)
                s["trend"] = s.y.rolling(14, min_periods=1).mean()
                fig = px.line(s, x="x", y=["y", "trend"], title=f"{name} trend (last 1000 records)")
                fig.update_layout(height=300, legend_title="", margin=dict(l=5, r=5, t=40, b=5))
                col.plotly_chart(fig)
            else:
                col.info(f"No {name.lower()} column in dataset.")
    with tabs[1]:
        sample = flood_df[FM["features"]].sample(min(len(flood_df), 3000), random_state=0)
        probs = FM["model"].predict_proba(sample)[:, 1]
        from src.flood_prediction import risk_level
        dist = pd.Series([risk_level(x) for x in probs]).value_counts().reindex(["LOW", "MODERATE", "HIGH", "EXTREME"]).fillna(0)
        c1, c2 = st.columns(2)
        c1.plotly_chart(px.bar(x=dist.index, y=dist.values, color=dist.index, color_discrete_map=LEVEL_COLORS,
                               title="Flood risk distribution (model on dataset sample)", labels={"x": "Risk", "y": "Count"}),
                        )
        c2.plotly_chart(px.histogram(x=probs, nbins=30, title="Predicted flood probability histogram", labels={"x": "Probability"}),
                        )
    with tabs[2]:
        if QI:
            q = QI.df.copy()
            c1, c2 = st.columns(2)
            c1.plotly_chart(px.histogram(q, x="mag", nbins=30, title="Historical earthquake magnitudes", labels={"mag": "Magnitude"}),
                            )
            if "date" in q and q.date.notna().any():
                yr = q.groupby(q.date.dt.year).size().reset_index(name="events")
                c2.plotly_chart(px.bar(yr, x="date", y="events", title="Earthquake events per year"))
        else:
            st.info("No earthquake catalogue (magnitude + latitude + longitude) found.")
        tgt = FM["target"]
        yb = flood_df[tgt]
        st.write(f"Flood label column used: **{tgt}** | positive-class rate in training data: **{FM['positive_rate']*100:.1f}%**")
    with tabs[3]:
        c1, c2 = st.columns(2)
        imp = FM["importance"].head(10)[::-1]
        c1.plotly_chart(px.bar(x=imp.values, y=imp.index, orientation="h", title="Feature importance (permutation, best model)",
                               labels={"x": "Importance", "y": ""}))
        c2.plotly_chart(px.bar(FM["metrics"], x="Model", y=["Accuracy", "F1 Score", "ROC-AUC"], barmode="group",
                               title="Model comparison (held-out test set)"))
        st.success(f"Best model selected: **{FM['best_name']}** (by ROC-AUC)")

# ================================================================== FLOOD PREDICTION
elif page == "🌊 Flood Prediction":
    st.title("🌊 Flood Risk Prediction & Live / Manual Analysis")
    if FM is None:
        st.stop()
    st.markdown("<div class='demo'>🎬 <b>Demo scenarios</b> - fill inputs from the dataset's own value ranges (not real-time data).</div>",
                unsafe_allow_html=True)
    b = st.columns(3)
    b[0].button("🌧️ Scenario 1: Heavy rain + high water level", on_click=heavy_rain)
    b[1].button("☀️ Scenario 2: Normal conditions", on_click=normal_cond)
    b[2].button("🌎 Scenario 3: Seismic demo → Earthquake page",
                on_click=lambda: st.session_state.update(goto_quake=True))
    if st.session_state.get("goto_quake"):
        st.info("Open **🌎 Earthquake Risk** in the sidebar and press the demo button there.")
    st.subheader("Enter environmental conditions")
    vals = feature_inputs("m_")
    if st.button("🔎 Run AI Analysis", type="primary") or True:
        p, lvl = predict_flood(FM, vals)
        qa = None
        if QI:
            lat = vals.get(fschema.get("lat")) if fschema.get("lat") in vals else None
            lon = vals.get(fschema.get("lon")) if fschema.get("lon") in vals else None
            if lat is not None and lon is not None:
                qa = QI.assess(float(lat), float(lon))
        ov = overall_alert(p, qa["score"] if qa else None)
        st.markdown("---")
        st.markdown(banner_html(ov["level"], f"{'FLOOD ' if ov['flood']==ov['level'] else ''}{ov['level']} ALERT"),
                    unsafe_allow_html=True)
        c = st.columns([1, 1, 1])
        c[0].plotly_chart(gauge(p, f"🌧️ Flood Risk: {lvl}", LEVEL_COLORS[lvl]))
        if qa:
            c[1].plotly_chart(gauge(qa["score"], f"🌎 Quake Indicator: {qa['level']}", LEVEL_COLORS[qa["level"]]),
                              )
        else:
            c[1].info("Earthquake indicator needs latitude & longitude inputs and a historical catalogue.")
        c[2].markdown(card("Overall Disaster Status", f"{STYLE[ov['level']][0]} {ov['level']}"), unsafe_allow_html=True)
        expl = explain_flood(FM, vals)
        t1, t2, t3 = st.tabs(["🔍 Why this prediction?", "✅ Recommended precautions", "📋 Detail table"])
        with t1:
            show_why(expl, lvl)
        with t2:
            st.markdown(f"**Flood actions for {ov['flood']}:**")
            for i, a in enumerate(FLOOD_ACTIONS[ov["flood"]], 1):
                st.markdown(f"{i}. {a}")
            if qa:
                st.markdown("**Earthquake preparedness:**")
                for a in QUAKE_ACTIONS:
                    st.markdown(f"- {a}")
            st.warning("In-app alert only - no SMS/email is sent. Follow official government/local disaster-management instructions.")
        with t3:
            st.dataframe(expl)

# ================================================================== EARTHQUAKE
elif page == "🌎 Earthquake Risk":
    st.title("🌎 Earthquake Risk Indicator & Safety")
    st.markdown(f"<div class='disc'>⚠️ {QUAKE_DISCLAIMER}</div>", unsafe_allow_html=True)
    t1, t2 = st.tabs(["Risk Indicator", "🌎 Earthquake Safety"])
    with t1:
        if QI is None:
            st.warning("No usable earthquake catalogue was found (needs magnitude, latitude and longitude columns). "
                       "Weather variables cannot predict earthquakes, so no indicator is produced. Add a seismic catalogue to data/.")
            if qi_err:
                st.caption(qi_err)
        else:
            if "q_lat" not in st.session_state:
                st.session_state.q_lat, st.session_state.q_lon = 23.5, 70.5
            def demo_q():
                st.session_state.q_lat, st.session_state.q_lon = QI.hotspot()
            st.button("🎬 Scenario 3: Seismic demo (densest historical cluster)", on_click=demo_q)
            c = st.columns(3)
            lat = c[0].number_input("Latitude", -90.0, 90.0, key="q_lat", format="%.3f")
            lon = c[1].number_input("Longitude", -180.0, 180.0, key="q_lon", format="%.3f")
            rad = c[2].slider("Search radius (km)", 50, 1000, 300, 50)
            QI.radius = rad
            qa = QI.assess(lat, lon)
            lvl = {"LOW": "NORMAL", "MODERATE": "WATCH", "HIGH": "WARNING"}[qa["level"]]
            st.markdown(banner_html(lvl, f"EARTHQUAKE RISK INDICATOR: {qa['level']}  ({qa['score']*100:.0f}%)",
                                    "Historical-catalogue based indicator, not a prediction."), unsafe_allow_html=True)
            c1, c2 = st.columns([1, 1])
            c1.plotly_chart(gauge(qa["score"], "Risk indicator", LEVEL_COLORS[qa["level"]]))
            with c2:
                st.markdown("**Historical/seismic factors considered:**")
                for f in qa["factors"]:
                    st.markdown(f"- {f}")
                st.caption("Method: 50% local event-density percentile + 30% largest nearby magnitude + 20% shallow-event share. "
                           "This is a transparent heuristic, not a trained forecasting model.")
            st.error(QUAKE_DISCLAIMER)
            if info["synthetic_quake"]:
                st.info("🟣 Catalogue shown is SYNTHETIC demo data.")
    with t2:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### During an Earthquake")
            st.markdown("- **DROP** to the ground\n- **COVER** your head and neck\n- **HOLD ON** to sturdy shelter\n"
                        "- Stay away from windows and falling objects\n- Do not use elevators\n"
                        "- Outdoors: move away from buildings, trees and power lines\n- Driving: stop safely and remain inside the vehicle")
        with c2:
            st.markdown("### After an Earthquake")
            st.markdown("- Check for injuries\n- Expect aftershocks\n- Avoid damaged buildings\n"
                        "- Check for gas/electrical hazards if safe\n- Follow official instructions")

# ================================================================== ASSISTANT
elif page == "🤖 Disaster AI Assistant":
    st.title("🤖 Disaster AI Assistant")
    st.caption(("🟢 LLM mode active" if llm_available() else
                "🟣 DEMO/FALLBACK mode: offline safety knowledge base (set ANTHROPIC_API_KEY to enable an LLM)"))
    st.markdown("<div class='disc'>I am not an emergency service. For location-specific emergencies follow official local "
                "disaster-management alerts and call your local emergency number.</div>", unsafe_allow_html=True)
    if "chat" not in st.session_state:
        st.session_state.chat = [{"role": "assistant", "content": "Hello! Ask me about earthquake or flood safety."}]
    quick = ["What should I do during an earthquake?", "What should I do during a flood?", "What should I keep in an emergency kit?",
             "Should I evacuate?", "What should I do if water enters my house?", "How can I protect children and elderly people?"]
    cols = st.columns(3)
    clicked = None
    for i, qn in enumerate(quick):
        if cols[i % 3].button(qn, key=f"q{i}"):
            clicked = qn
    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
    typed = st.chat_input("Ask a safety question...")
    msg = clicked or typed
    if msg:
        st.session_state.chat.append({"role": "user", "content": msg})
        ctx = ""
        reply = answer(msg, st.session_state.chat[:-1], ctx)
        st.session_state.chat.append({"role": "assistant", "content": reply})
        st.rerun()

# ================================================================== EMERGENCY GUIDE
elif page == "🚨 Emergency Guide":
    st.title("🚨 Emergency Guide")
    st.markdown(f"<div class='disc'>{DISCLAIMER}</div>", unsafe_allow_html=True)
    t1, t2, t3 = st.tabs(["🌊 Flood Safety", "🌎 Earthquake Safety", "📞 Emergency Contacts"])
    with t1:
        with st.expander("Before a Flood", True):
            st.markdown("- Monitor official warnings\n- Prepare an emergency kit\n- Keep important documents protected\n- Charge phones/power banks\n- Know evacuation routes")
        with st.expander("During a Flood", True):
            st.markdown("- Move to higher ground when advised\n- Never walk or drive through floodwater\n- Stay away from electrical equipment and downed power lines\n- Follow evacuation orders\n- Keep emergency communication available")
        with st.expander("After a Flood", True):
            st.markdown("- Avoid contaminated water\n- Do not enter damaged buildings until declared safe\n- Watch for electrical hazards\n- Follow local authority instructions")
    with t2:
        with st.expander("During an Earthquake", True):
            st.markdown("- **DROP**, **COVER**, **HOLD ON**\n- Stay away from windows and falling objects\n- Do not use elevators\n- Outdoors: move away from buildings, trees and power lines\n- Driving: stop safely and remain inside")
        with st.expander("After an Earthquake", True):
            st.markdown("- Check for injuries\n- Expect aftershocks\n- Avoid damaged buildings\n- Check gas/electrical hazards if safe\n- Follow official instructions")
    with t3:
        st.info("Numbers are NOT pre-filled. Enter your region's official numbers - they are stored locally in data/emergency_contacts.json.")
        contacts = load_contacts()
        icons = {"Local Emergency Services": "🚨", "Police": "🚓", "Ambulance": "🚑", "Fire & Rescue": "🚒", "Disaster Management Authority": "🌐"}
        new = {}
        for k, v in contacts.items():
            new[k] = st.text_input(f"{icons.get(k, '☎️')} {k}", value=v, placeholder="Enter number / website")
        if st.button("💾 Save contacts"):
            try:
                with open(CONTACT_FILE, "w") as f:
                    json.dump(new, f, indent=2)
                st.success("Saved.")
            except Exception as e:
                st.error(f"Could not save: {e}")

# ================================================================== DATA ANALYSIS
elif page == "📊 Data Analysis":
    st.title("📊 Data Analysis")
    st.write(f"**Flood dataset source:** {info['flood_source']}")
    if info["files"]:
        with st.expander("Files found in data/"):
            st.write({k: f"{v[0]} rows × {v[1]} cols" for k, v in info["files"].items()})
    d = describe_dataset(flood_df)
    c = st.columns(4)
    c[0].metric("Records", f"{d['rows']:,}"); c[1].metric("Columns", d["cols"])
    c[2].metric("Numeric", len(d["numeric"])); c[3].metric("Categorical", len(d["categorical"]))
    st.subheader("Detected column roles")
    st.json({**fschema, "flood_target": detect_flood_target(flood_df)})
    tabs = st.tabs(["Preview", "Missing values", "Statistics", "Correlation", "Distributions", "Target & model metrics"])
    with tabs[0]:
        st.dataframe(flood_df.head(100))
    with tabs[1]:
        if len(d["missing"]):
            st.dataframe(d["missing"].rename("missing").to_frame())
            st.caption("Missing numeric values are imputed with the median inside the model pipeline.")
        else:
            st.success("No missing values.")
    with tabs[2]:
        st.dataframe(flood_df.describe().T)
    with tabs[3]:
        num = flood_df.select_dtypes(include=np.number)
        if num.shape[1] > 1:
            st.plotly_chart(px.imshow(num.corr(), text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto"),
                            )
    with tabs[4]:
        num = flood_df.select_dtypes(include=np.number).columns.tolist()
        sel = st.selectbox("Column", num)
        st.plotly_chart(px.histogram(flood_df, x=sel, nbins=40, marginal="box"))
    with tabs[5]:
        if FM:
            st.write(f"Target column: **{FM['target']}** → binary label. Positive rate: **{FM['positive_rate']*100:.1f}%** | "
                     f"train {FM['n_train']:,} / test {FM['n_test']:,}")
            tv = flood_df[FM["target"]].value_counts().reset_index()
            tv.columns = ["value", "count"]
            tv["value"] = tv["value"].astype(str)
            st.plotly_chart(px.bar(tv.head(20), x="value", y="count", title="Target distribution"))
            st.subheader("Model evaluation (held-out test set)")
            st.dataframe(FM["metrics"].style.format({c: "{:.3f}" for c in FM["metrics"].columns[1:]}))
            cm = FM["confusion"]
            st.plotly_chart(px.imshow(cm, text_auto=True, x=["Pred No", "Pred Flood"], y=["Actual No", "Actual Flood"],
                                      title=f"Confusion matrix - {FM['best_name']}", color_continuous_scale="Blues"),
                            )
            if info["synthetic_flood"]:
                st.warning("These metrics are for SYNTHETIC data whose labels follow a made-up rule - they do not reflect real-world accuracy.")
            st.caption("Retrain: replace the file in data/ (or upload in sidebar) and the models are retrained automatically. "
                       "Saved model: models/flood_model.pkl")

# ================================================================== MAP
elif page == "🗺️ Risk Map":
    st.title("🗺️ Risk Map")
    layers = []
    if "lat" in fschema and "lon" in fschema and FM:
        base = flood_df.dropna(subset=[fschema["lat"], fschema["lon"]])
        base = base.sample(min(len(base), 1500), random_state=0)
        from src.flood_prediction import risk_level
        d = pd.DataFrame({"lat": pd.to_numeric(base[fschema["lat"]], errors="coerce").values,
                          "lon": pd.to_numeric(base[fschema["lon"]], errors="coerce").values})
        d["Flood probability"] = FM["model"].predict_proba(base[FM["features"]])[:, 1]
        d["Flood risk"] = d["Flood probability"].apply(risk_level)
        d = d.dropna(subset=["lat", "lon"])
        layers.append(("flood", d))
    show_f = st.checkbox("Flood-risk locations (model output)", True)
    show_q = st.checkbox("Historical earthquakes", True)
    NEW_MAP = hasattr(go, "Scattermap")
    MAPTRACE = go.Scattermap if NEW_MAP else go.Scattermapbox
    fig = go.Figure()
    if show_f and layers:
        d = layers[0][1]
        for lv, col in LEVEL_COLORS.items():
            s = d[d["Flood risk"] == lv]
            if len(s):
                fig.add_trace(MAPTRACE(lat=s.lat, lon=s.lon, mode="markers", marker=dict(size=7, color=col),
                                               name=f"Flood: {lv}", text=s["Flood probability"].round(2).astype(str)))
    if show_q and QI:
        q = QI.df.sample(min(len(QI.df), 1500), random_state=0)
        fig.add_trace(MAPTRACE(lat=q.lat, lon=q.lon, mode="markers", name="Historical earthquakes",
                                       marker=dict(size=(q.mag - 2).clip(lower=2) * 2.2, color="#7c3aed", opacity=0.5),
                                       text=("M" + q.mag.round(1).astype(str))))
    if not fig.data:
        st.warning("No latitude/longitude information available in the dataset - map cannot be shown.")
    else:
        allpts = pd.concat([pd.DataFrame({"lat": t.lat, "lon": t.lon}) for t in fig.data])
        mp = dict(style="carto-positron", center=dict(lat=float(allpts.lat.mean()), lon=float(allpts.lon.mean())), zoom=4)
        fig.update_layout(**({"map": mp} if NEW_MAP else {"mapbox": mp}), height=620, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig)
        if info["synthetic_flood"] or info["synthetic_quake"]:
            st.info("🟣 Coordinates in demo mode are SYNTHETIC - they do not represent real flood or earthquake locations.")
        st.caption("Colours show model-estimated flood risk at dataset points; purple circles are catalogued historical earthquakes.")

# ================================================================== ABOUT
elif page == "ℹ️ About Project":
    st.title("ℹ️ About Project")
    st.subheader("AI Disaster Alert & Emergency Assistant")
    st.markdown("**Objective:** To develop an AI-powered disaster risk assessment and alert system that analyzes environmental and "
                "historical data to estimate flood risk, provide earthquake risk indicators, and deliver AI-assisted emergency "
                "preparedness guidance.")
    st.markdown("**Technologies:** Python · Machine Learning · Streamlit · Pandas · NumPy · Scikit-learn · Plotly · "
                "Explainable AI (permutation importance & occlusion) · Generative AI / AI Assistant (optional LLM)")
    st.markdown("**Modules:** `preprocessing` (auto schema detection) · `flood_prediction` (multi-model comparison) · "
                "`earthquake_risk` (historical risk indicator) · `alert_engine` · `assistant`")
    st.subheader("Limitations")
    st.markdown("- Earthquakes cannot be predicted; the indicator is a historical heuristic.\n- Flood accuracy depends on dataset quality/coverage.\n"
                "- No live sensor/weather API; no real SMS/email (in-app alerts only).\n- Fallback assistant is rule-based unless an API key is set.")
    st.subheader("Future scope")
    st.markdown("- Live weather/river-gauge APIs\n- SHAP & time-series (LSTM) models\n- Real notification channels (SMS/email/WhatsApp)\n- Multilingual assistant\n- Satellite/DEM-based flood mapping")
    st.markdown(f"<div class='disc'>{DISCLAIMER}</div>", unsafe_allow_html=True)
