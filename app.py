
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import base64, io, joblib

# ===============================
# LOAD MODEL FROM SECRETS
# ===============================
model_bytes = base64.b64decode(st.secrets["rf_collision_model.pkl"])
model = joblib.load(io.BytesIO(model_bytes))

# ===============================
# ORBIT FUNCTIONS (SAME AS TRAINING)
# ===============================
def rotate_z(vec, angle):
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s, 0],[s, c, 0],[0,0,1]])
    return R @ vec

def rotate_x(vec, angle):
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[1,0,0],[0,c,-s],[0,s,c]])
    return R @ vec

def orbital_radius(a, e, theta):
    return a * (1 - e**2) / (1 + e * np.cos(theta))

def classical_to_eci(a, e, i_deg, raan_deg, argp_deg, theta):
    r = orbital_radius(a, e, theta)
    x_op = r * np.cos(theta)
    y_op = r * np.sin(theta)
    z_op = 0.0
    vec = np.array([x_op, y_op, z_op])
    vec = rotate_z(vec, np.deg2rad(argp_deg))
    vec = rotate_x(vec, np.deg2rad(i_deg))
    vec = rotate_z(vec, np.deg2rad(raan_deg))
    return vec

def generate_random_orbit():
    a = 6371 + np.random.uniform(200, 1200)
    e = np.random.beta(1, 8) * 0.05
    i = np.random.uniform(0, 90)
    raan = np.random.uniform(0, 360)
    argp = np.random.uniform(0, 360)
    return {"a": a, "e": e, "i": i, "raan": raan, "argp": argp}

def compute_features(o1, o2):
    theta0 = 0
    p1 = classical_to_eci(o1["a"], o1["e"], o1["i"], o1["raan"], o1["argp"], theta0)
    p2 = classical_to_eci(o2["a"], o2["e"], o2["i"], o2["raan"], o2["argp"], theta0)
    sep = np.linalg.norm(p1 - p2)
    alt1 = np.linalg.norm(p1) - 6371
    alt2 = np.linalg.norm(p2) - 6371
    alt_diff = abs(alt1 - alt2)
    a_diff = abs(o1["a"] - o2["a"])
    i_diff = abs(o1["i"] - o2["i"])
    mu = 398600.4418
    v1 = np.sqrt(mu * (2/np.linalg.norm(p1) - 1/o1["a"]))
    v2 = np.sqrt(mu * (2/np.linalg.norm(p2) - 1/o2["a"]))
    rel_speed = abs(v1 - v2)
    return np.array([sep, alt_diff, a_diff, i_diff, rel_speed]).reshape(1,-1)

def get_positions(o, steps=200):
    thetas = np.linspace(0, 2*np.pi, steps)
    return np.array([classical_to_eci(o["a"], o["e"], o["i"], o["raan"], o["argp"], t) for t in thetas])

# ===============================
# STREAMLIT APP LAYOUT
# ===============================
st.set_page_config(page_title="AI-Powered Space Junk Tracker", layout="wide")
st.title("🛰️ AI-Powered Space Junk Tracker")
st.write("Predicts potential satellite collision risk using ML.")

st.sidebar.header("Controls")
generate_button = st.sidebar.button("🔄 Generate New Pair")

# Keep orbits in session state so they persist between reruns
if "o1" not in st.session_state or generate_button:
    st.session_state.o1 = generate_random_orbit()
    st.session_state.o2 = generate_random_orbit()

o1 = st.session_state.o1
o2 = st.session_state.o2

# Run prediction
features = compute_features(o1, o2)
pred_prob = model.predict_proba(features)[0, 1]
pred_label = "🚨 COLLISION RISK" if pred_prob > 0.5 else "✅ SAFE PASS"

# Visualize in 3D
pos1 = get_positions(o1, 300)
pos2 = get_positions(o2, 300)
fig = go.Figure()

# Earth Sphere
R = 6371
u = np.linspace(0, 2*np.pi, 40)
v = np.linspace(0, np.pi, 20)
x = R * np.outer(np.cos(u), np.sin(v))
y = R * np.outer(np.sin(u), np.sin(v))
z = R * np.outer(np.ones_like(u), np.cos(v))
fig.add_trace(go.Surface(x=x, y=y, z=z, opacity=0.7, showscale=False))

# Orbits
fig.add_trace(go.Scatter3d(x=pos1[:,0], y=pos1[:,1], z=pos1[:,2], mode="lines", name="Orbit 1"))
fig.add_trace(go.Scatter3d(x=pos2[:,0], y=pos2[:,1], z=pos2[:,2], mode="lines", name="Orbit 2"))

fig.update_layout(scene=dict(aspectmode="data"), title=f"Prediction: {pred_label} (Prob={pred_prob:.2f})")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Prediction Result")
st.metric(label="Collision Probability", value=f"{pred_prob:.2%}", delta="HIGH" if pred_prob > 0.5 else "LOW")
