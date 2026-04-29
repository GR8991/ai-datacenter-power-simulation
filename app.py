import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.ai_load_profile import AILoadProfile
from models.gas_generator import GasGenerator
from models.bess_model import BESSModel
from simulation.engine import SimulationEngine

# ─── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="AI Data Center Power Simulation",
    page_icon="⚡",
    layout="wide"
)

# ─── Title ─────────────────────────────────────────────────
st.title("⚡ AI Data Center — Power Simulation Dashboard")
st.markdown("**Based on NVIDIA BESS Self-Qualification Guidelines v0.4**")
st.divider()

# ─── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Simulation Config")

    st.subheader("🖥️ AI Data Center")
    it_load_mw     = st.slider("Total IT Load (MW)", 10, 500, 100)
    num_gpus       = st.slider("Number of GPUs", 1000, 50000, 8192)
    gpu_power_kw   = st.selectbox("GPU Type (Max Power)", 
                                   ["H100 (700W)", "A100 (400W)", "H200 (1000W)"])
    ramp_rate      = st.slider("Ramp Rate (% IT Load/sec)", 5, 50, 20,
                                help="NVIDIA requires 20% IT load/second")
    sim_duration   = st.slider("Simulation Duration (minutes)", 5, 120, 30)

    st.subheader("⛽ Gas Generator")
    gen_rated_mw   = st.slider("Generator Rated Power (MW)", 10, 600, 120)
    droop_pct      = st.slider("Droop Setting (%)", 1, 10, 5)
    governor_speed = st.slider("Governor Response Time (sec)", 1, 10, 3)
    min_load_pct   = st.slider("Minimum Load (%)", 10, 40, 25,
                                help="Below this, wet stacking risk")

    st.subheader("🔋 BESS")
    bess_capacity  = st.slider("BESS Capacity (MWh)", 1, 100, 20)
    bess_power     = st.slider("BESS Max Power (MW)", 1, 200, 50)
    soc_initial    = st.slider("Initial SOC (%)", 20, 100, 80)
    soc_min        = st.slider("Min SOC (%)", 5, 30, 20)
    soc_max        = st.slider("Max SOC (%)", 70, 100, 90)
    bess_mode      = st.selectbox("BESS Mode", 
                                   ["Grid-Forming (GFM)", "Grid-Following (GFL)"])

    st.subheader("🧱 Dummy Load Bank")
    dummy_load_mw  = st.slider("Dummy Load Bank Capacity (MW)", 0, 100, 30)

    st.divider()
    run_sim = st.button("▶️ RUN SIMULATION", type="primary", use_container_width=True)

# ─── Tabs ──────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 AI Load Profile",
    "⛽ Generator Response",
    "🔋 BESS Behavior",
    "⚖️ BESS vs Dummy Load",
    "✅ NVIDIA Test Results"
])

if run_sim:
    with st.spinner("Running simulation..."):

        # Parse GPU power
        gpu_map = {"H100 (700W)": 700, "A100 (400W)": 400, "H200 (1000W)": 1000}
        gpu_power_w = gpu_map[gpu_power_kw]

        # Initialize models
        load_model = AILoadProfile(
            it_load_mw=it_load_mw,
            num_gpus=num_gpus,
            gpu_power_w=gpu_power_w,
            ramp_rate_pct=ramp_rate,
            duration_min=sim_duration
        )

        gen_model = GasGenerator(
            rated_mw=gen_rated_mw,
            droop_pct=droop_pct,
            governor_time_s=governor_speed,
            min_load_pct=min_load_pct
        )

        bess_model = BESSModel(
            capacity_mwh=bess_capacity,
            max_power_mw=bess_power,
            soc_initial_pct=soc_initial,
            soc_min_pct=soc_min,
            soc_max_pct=soc_max,
            mode=bess_mode,
            dummy_load_mw=dummy_load_mw
        )

        # Run simulation engine
        engine = SimulationEngine(load_model, gen_model, bess_model)
        results = engine.run()

        df = results["timeseries"]
        tests = results["nvidia_tests"]

    # ── Tab 1: AI Load Profile ──────────────────────────────
    with tab1:
        st.subheader("📊 AI Workload Power Demand Profile")
        st.caption("Simulates GPU workload ramps based on NVIDIA Test 4 — 20% IT load/second")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Peak Load", f"{df['it_load_mw'].max():.1f} MW")
        col2.metric("Min Load",  f"{df['it_load_mw'].min():.1f} MW")
        col3.metric("Avg Load",  f"{df['it_load_mw'].mean():.1f} MW")
        col4.metric("Load Drop Events", 
                    f"{int(results['num_drop_events'])}")

        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=df["time_s"], y=df["it_load_mw"],
            name="IT Load (MW)", line=dict(color="#00D4FF", width=2),
            fill="tozeroy", fillcolor="rgba(0,212,255,0.1)"
        ))

        # Highlight sudden drop events
        for event in results["drop_events"]:
            fig1.add_vrect(
                x0=event["start"], x1=event["end"],
                fillcolor="rgba(255,50,50,0.2)",
                annotation_text="⚠️ Load Drop",
                annotation_position="top left",
                line_width=0
            )

        fig1.update_layout(
            title="AI Workload Power Profile",
            xaxis_title="Time (seconds)",
            yaxis_title="Power (MW)",
            template="plotly_dark",
            height=400
        )
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("📋 Load Event Table")
        st.dataframe(pd.DataFrame(results["drop_events"]), use_container_width=True)

    # ── Tab 2: Generator Response ───────────────────────────
    with tab2:
        st.subheader("⛽ Gas Generator Response to Load Changes")

        col1, col2, col3 = st.columns(3)
        col1.metric("Max Frequency Deviation",
                    f"{df['freq_deviation_hz'].abs().max():.3f} Hz",
                    delta="Normal < 0.5 Hz")
        col2.metric("Overspeed Events",
                    f"{int(results['overspeed_events'])}")
        col3.metric("Generator Utilization",
                    f"{df['gen_output_mw'].mean() / gen_rated_mw * 100:.1f}%")

        fig2 = make_subplots(rows=2, cols=1,
                             subplot_titles=("Generator Output (MW)",
                                             "Frequency Deviation (Hz)"))

        fig2.add_trace(go.Scatter(
            x=df["time_s"], y=df["gen_output_mw"],
            name="Generator Output", line=dict(color="#FFB300", width=2)
        ), row=1, col=1)

        # Min load threshold line
        fig2.add_hline(
            y=gen_rated_mw * min_load_pct / 100,
            line_dash="dash", line_color="red",
            annotation_text="Min Load Threshold (wet stacking risk)",
            row=1, col=1
        )

        fig2.add_trace(go.Scatter(
            x=df["time_s"], y=df["freq_deviation_hz"],
            name="Frequency Deviation", 
            line=dict(color="#FF5733", width=2)
        ), row=2, col=1)

        fig2.add_hline(y=0.5,  line_dash="dash",
                       line_color="orange",
                       annotation_text="Warning ±0.5 Hz", row=2, col=1)
        fig2.add_hline(y=-0.5, line_dash="dash",
                       line_color="orange", row=2, col=1)

        fig2.update_layout(template="plotly_dark", height=600)
        st.plotly_chart(fig2, use_container_width=True)

    # ── Tab 3: BESS Behavior ────────────────────────────────
    with tab3:
        st.subheader("🔋 BESS State of Charge & Power Flow")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Final SOC",       f"{df['soc_pct'].iloc[-1]:.1f}%")
        col2.metric("SOC Drift",       f"{results['soc_drift']:.2f}%",
                    delta="Pass if < 5%" )
        col3.metric("Energy Absorbed", f"{results['energy_absorbed_mwh']:.2f} MWh")
        col4.metric("Energy Saved vs Dummy Load",
                    f"${results['cost_saving_usd']:,.0f}")

        fig3 = make_subplots(rows=2, cols=1,
                             subplot_titles=("SOC Over Time (%)",
                                             "BESS Power Flow (MW)"))

        fig3.add_trace(go.Scatter(
            x=df["time_s"], y=df["soc_pct"],
            name="SOC", line=dict(color="#00FF88", width=2),
            fill="tozeroy", fillcolor="rgba(0,255,136,0.1)"
        ), row=1, col=1)

        fig3.add_hline(y=soc_max, line_dash="dash",
                       line_color="red",
                       annotation_text=f"SOC Max {soc_max}%", row=1, col=1)
        fig3.add_hline(y=soc_min, line_dash="dash",
                       line_color="orange",
                       annotation_text=f"SOC Min {soc_min}%", row=1, col=1)

        fig3.add_trace(go.Scatter(
            x=df["time_s"], y=df["bess_power_mw"],
            name="BESS Power (+charge / -discharge)",
            line=dict(color="#BB86FC", width=2)
        ), row=2, col=1)

        fig3.add_hline(y=0, line_color="white",
                       line_dash="dot", row=2, col=1)

        fig3.update_layout(template="plotly_dark", height=600)
        st.plotly_chart(fig3, use_container_width=True)

    # ── Tab 4: BESS vs Dummy Load ───────────────────────────
    with tab4:
        st.subheader("⚖️ BESS Charging vs Dummy Load Bank Comparison")
        st.caption("Shows how much surplus power went to BESS charging vs wasted as heat")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Energy stored in BESS",
                      f"{results['energy_absorbed_mwh']:.2f} MWh", "✅ Saved")
            st.metric("Energy wasted in Dummy Load",
                      f"{results['dummy_load_mwh']:.2f} MWh", "❌ Wasted as heat")

        with col2:
            # Pie chart
            fig_pie = go.Figure(go.Pie(
                labels=["BESS Absorbed", "Dummy Load (Wasted)", "Generator Reduced"],
                values=[
                    results["energy_absorbed_mwh"],
                    results["dummy_load_mwh"],
                    results["gen_reduced_mwh"]
                ],
                marker_colors=["#00FF88", "#FF5733", "#FFB300"],
                hole=0.4
            ))
            fig_pie.update_layout(
                title="Surplus Power Distribution",
                template="plotly_dark", height=350
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        # Stacked area chart
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(
            x=df["time_s"], y=df["bess_charging_mw"],
            name="BESS Charging (MW)", stackgroup="one",
            line=dict(color="#00FF88"),
            fillcolor="rgba(0,255,136,0.4)"
        ))
        fig4.add_trace(go.Scatter(
            x=df["time_s"], y=df["dummy_load_active_mw"],
            name="Dummy Load Active (MW)", stackgroup="one",
            line=dict(color="#FF5733"),
            fillcolor="rgba(255,87,51,0.4)"
        ))
        fig4.update_layout(
            title="Surplus Power Absorption Over Time",
            xaxis_title="Time (seconds)",
            yaxis_title="Power (MW)",
            template="plotly_dark", height=400
        )
        st.plotly_chart(fig4, use_container_width=True)

    # ── Tab 5: NVIDIA Test Results ──────────────────────────
    with tab5:
        st.subheader("✅ NVIDIA BESS Qualification Test Results")
        st.caption("Based on NVIDIA BESS Self-Qualification Guidelines v0.4 — February 2026")

        for test_name, test_result in tests.items():
            status  = test_result["status"]
            icon    = "✅" if status == "PASS" else "❌"
            color   = "green" if status == "PASS" else "red"

            with st.expander(f"{icon} {test_name} — **{status}**"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Criteria:**")
                    for k, v in test_result["criteria"].items():
                        st.write(f"• {k}: `{v}`")
                with col2:
                    st.markdown("**Measured:**")
                    for k, v in test_result["measured"].items():
                        measured_ok = test_result.get("details", {}).get(k, True)
                        tick = "✅" if measured_ok else "❌"
                        st.write(f"{tick} {k}: `{v}`")

                if "recommendation" in test_result:
                    st.warning(f"💡 {test_result['recommendation']}")

else:
    # Show placeholder before simulation runs
    with tab1:
        st.info("👈 Configure parameters in the sidebar and click **▶️ RUN SIMULATION**")
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Gas_turbine_Wärtsilä.jpg/320px-Gas_turbine_Wärtsilä.jpg",
                 caption="AI Data Center Gas Generator + BESS System")
    with tab2:
        st.info("👈 Run simulation to see generator response")
    with tab3:
        st.info("👈 Run simulation to see BESS behavior")
    with tab4:
        st.info("👈 Run simulation to see BESS vs Dummy Load comparison")
    with tab5:
        st.info("👈 Run simulation to see NVIDIA qualification test results")
