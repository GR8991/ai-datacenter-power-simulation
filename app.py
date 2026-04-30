import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.ai_load_profile import AILoadProfile
from models.raps_loader import RAPSLoader
from models.gas_generator import GasGenerator
from models.bess_model import BESSModel
from simulation.engine import SimulationEngine

# ── Page Config ────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Data Center Power Simulation",
    page_icon="⚡",
    layout="wide"
)

# ── Title ──────────────────────────────────────────────────────
st.title("⚡ AI Data Center — Power Simulation Dashboard")
st.markdown(
    "**Based on NVIDIA BESS Self-Qualification Guidelines v0.4 "
    "(February 2026)**"
)
st.info(
    "📌 **Simulation Note:** This tool supports both synthetic "
    "AI workload profiles AND real ExaDigiT RAPS job trace data. "
    "Based on NVIDIA BESS Qualification parameters "
    "(Test 2, 4, 9, 11). Results are for educational "
    "and planning purposes only."
)
st.divider()

# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Simulation Config")

    st.subheader("📂 Data Source")
    data_source = st.radio(
        "Select Workload Source",
        ["🔢 Synthetic (Phase 1)", "📂 Real RAPS Data (Phase 2)"],
        index=0
    )

    st.subheader("🖥️ AI Data Center")
    it_load_mw     = st.slider("Total IT Load (MW)", 10, 500, 100)
    num_gpus       = st.slider("Number of GPUs", 1000, 50000, 8192)
    gpu_power_kw   = st.selectbox(
        "GPU Type (Max Power)",
        ["H100 (700W)", "A100 (400W)", "H200 (1000W)"]
    )
    ramp_rate      = st.slider(
        "Ramp Rate (% IT Load/sec)", 5, 50, 20,
        help="NVIDIA PERF-CORE-01 requires <= 20% IT load/second"
    )
    sim_duration   = st.slider(
        "Simulation Duration (minutes)", 5, 120, 30
    )

    st.subheader("⛽ Gas Generator")
    gen_rated_mw   = st.slider("Generator Rated Power (MW)", 10, 600, 120)
    droop_pct      = st.slider("Droop Setting (%)", 1, 10, 5)
    governor_speed = st.slider("Governor Response Time (sec)", 1, 10, 3)
    min_load_pct   = st.slider(
        "Minimum Load (%)", 10, 40, 25,
        help="Below this threshold, wet stacking risk increases"
    )

    st.subheader("🔋 BESS")
    bess_capacity  = st.slider("BESS Capacity (MWh)", 1, 100, 20)
    bess_power     = st.slider("BESS Max Power (MW)", 1, 200, 50)
    soc_initial    = st.slider("Initial SOC (%)", 20, 100, 80)
    soc_min        = st.slider("Min SOC (%)", 5, 30, 20)
    soc_max        = st.slider("Max SOC (%)", 70, 100, 90)
    bess_mode      = st.selectbox(
        "BESS Mode",
        ["Grid-Forming (GFM)", "Grid-Following (GFL)"]
    )

    st.subheader("🧱 Dummy Load Bank")
    dummy_load_mw  = st.slider(
        "Dummy Load Bank Capacity (MW)", 0, 100, 30
    )

    st.divider()
    run_sim = st.button(
        "▶️ RUN SIMULATION",
        type="primary",
        use_container_width=True
    )

    st.divider()
    st.markdown("**📖 References**")
    st.markdown("[NVIDIA BESS Guidelines v0.4](https://docs.nvidia.com/datacenter/dsx/BESS-Self-Qualification-Guidelines.html)")
    st.markdown("[ExaDigiT RAPS](https://code.ornl.gov/exadigit/raps)")
    st.markdown("[EPRI DCFlex](https://dcflex.epri.com/)")
    st.markdown("[Zenodo Dataset](https://zenodo.org/records/10127767)")

# ── Tabs ───────────────────────────────────────────────────────
tab0, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📂 RAPS Data",
    "📊 AI Load Profile",
    "⛽ Generator Response",
    "🔋 BESS Behavior",
    "⚖️ BESS vs Dummy Load",
    "✅ NVIDIA Test Results",
    "ℹ️ About"
])

# ── Tab 0: RAPS Data ───────────────────────────────────────────
with tab0:
    st.subheader("📂 ExaDigiT RAPS Data Integration")
    st.markdown(
        "**Phase 2A — Real HPC Workload Data "
        "from Oak Ridge National Lab**"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### What is RAPS?")
        st.markdown(
            "RAPS (Resource Allocator and Power Simulator) is the "
            "official ExaDigiT simulation tool from Oak Ridge "
            "National Laboratory. It provides real supercomputer "
            "job traces that generate realistic AI workload "
            "power profiles — directly replacing synthetic data."
        )
        st.markdown("### How It Works — No Upload Needed!")
        st.markdown("**Step 1:** Select **Real RAPS Data** in sidebar")
        st.markdown("**Step 2:** Click **Load RAPS Data** button below")
        st.markdown(
            "**Step 3:** Data downloads automatically from Zenodo"
        )
        st.markdown("**Step 4:** Data cached — no re-download needed")
        st.markdown("**Step 5:** Click **RUN SIMULATION**")
        st.success(
            "📥 Data fetched directly from Zenodo. "
            "No file upload required! "
            "Automatically cached after first download."
        )

    with col2:
        st.markdown("### Dataset Info")
        st.markdown("- **Source:** Marconi100 Supercomputer, CINECA Italy")
        st.markdown("- **Jobs:** Real HPC job scheduling traces")
        st.markdown("- **Format:** Apache Parquet (columnar)")
        st.markdown("- **Size:** ~270 MB")
        st.markdown("- **License:** Open access via Zenodo")
        st.markdown("- **DOI:** 10.5281/zenodo.10127767")

        st.markdown("### Direct URL")
        st.code(
            "https://zenodo.org/records/10127767"
            "/files/job_table.parquet?download=1",
            language=None
        )

        st.markdown("### Why Real Data Matters")
        st.markdown(
            "Real HPC job traces capture actual workload patterns — "
            "unpredictable start/stop times, varied GPU counts, "
            "and realistic power ramps that synthetic data "
            "cannot replicate."
        )

    st.divider()
    st.markdown("### Load RAPS Data")

    col_btn1, col_btn2, col_btn3 = st.columns(3)

    with col_btn1:
        load_raps_btn = st.button(
            "📥 Load RAPS Data from Zenodo",
            type="primary",
            use_container_width=True
        )
    with col_btn2:
        clear_btn = st.button(
            "🗑️ Clear Cached Data",
            use_container_width=True
        )
    with col_btn3:
        if "raps_loaded" in st.session_state:
            st.success("✅ RAPS data loaded & cached!")
        else:
            st.warning("⏳ Not loaded yet")

    # ── Clear cache ────────────────────────────────────────────
    if clear_btn:
        for key in ["raps_loaded", "raps_stats", "raps_cols"]:
            if key in st.session_state:
                del st.session_state[key]
        RAPSLoader.clear_cache()
        st.success("🗑️ Cache cleared! Click Load to re-download.")

    # ── Load RAPS data ─────────────────────────────────────────
    if load_raps_btn:
        try:
            with st.spinner(
                "📥 Downloading from Zenodo (~270MB)... "
                "This may take 1-2 minutes..."
            ):
                loader = RAPSLoader(
                    it_load_mw   = it_load_mw,
                    gpu_power_w  = 700,
                    duration_min = sim_duration
                )
                stats = loader.get_job_stats()
                cols  = loader.get_columns()

            st.session_state["raps_loaded"] = True
            st.session_state["raps_stats"]  = stats
            st.session_state["raps_cols"]   = cols

            st.success(
                "✅ RAPS data loaded successfully from Zenodo!"
            )

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Jobs",
                      f"{stats['total_jobs']:,}")
            c2.metric("Columns",
                      f"{len(stats['columns'])}")
            c3.metric(
                "Time Span",
                f"{stats['time_span_h']} hrs"
                if stats["time_span_h"] else "N/A"
            )
            c4.metric(
                "Max GPUs",
                f"{stats['max_gpus']:,}"
                if stats["max_gpus"] else "N/A"
            )

            st.markdown("### Dataset Columns Detected")
            st.code(", ".join(cols), language=None)

            st.markdown("### Data Preview (first 10 rows)")
            path, _ = RAPSLoader._download_data()
            df_prev = pd.read_parquet(path)
            st.dataframe(
                df_prev.head(10),
                use_container_width=True
            )

            st.markdown("### GPU Count Distribution")
            gpu_col = next(
                (c for c in ["num_gpus", "gpus", "nGPUs"]
                 if c in df_prev.columns), None
            )
            if gpu_col:
                fig_hist = go.Figure(go.Histogram(
                    x=df_prev[gpu_col],
                    nbinsx=50,
                    marker_color="#00D4FF",
                    opacity=0.8
                ))
                fig_hist.update_layout(
                    title="GPU Count Distribution Across Jobs",
                    xaxis_title="Number of GPUs per Job",
                    yaxis_title="Job Count",
                    template="plotly_dark",
                    height=350
                )
                st.plotly_chart(
                    fig_hist, use_container_width=True
                )

            st.markdown("### Job Duration Distribution")
            start_col = next(
                (c for c in
                 ["start_time", "start", "submit_time"]
                 if c in df_prev.columns), None
            )
            end_col = next(
                (c for c in
                 ["end_time", "end", "finish_time"]
                 if c in df_prev.columns), None
            )
            if start_col and end_col:
                try:
                    start_s = RAPSLoader._to_seconds(
                        df_prev[start_col]
                    )
                    end_s   = RAPSLoader._to_seconds(
                        df_prev[end_col]
                    )
                    duration_h = (end_s - start_s) / 3600
                    fig_dur = go.Figure(go.Histogram(
                        x=duration_h.clip(0, 48),
                        nbinsx=50,
                        marker_color="#FFB300",
                        opacity=0.8
                    ))
                    fig_dur.update_layout(
                        title="Job Duration Distribution (hours)",
                        xaxis_title="Duration (hours)",
                        yaxis_title="Job Count",
                        template="plotly_dark",
                        height=350
                    )
                    st.plotly_chart(
                        fig_dur, use_container_width=True
                    )
                except Exception:
                    pass

        except Exception as e:
            st.error(f"❌ Error loading RAPS data: {e}")
            st.warning(
                "💡 Possible reasons:\n"
                "- Streamlit Cloud outbound internet blocked\n"
                "- Zenodo temporarily unavailable\n"
                "- File size too large for memory\n\n"
                "Try switching to Synthetic data mode."
            )

    elif "raps_stats" in st.session_state:
        stats = st.session_state["raps_stats"]
        cols  = st.session_state["raps_cols"]

        st.info("📋 Showing cached RAPS data stats")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Jobs",
                  f"{stats['total_jobs']:,}")
        c2.metric("Columns",
                  f"{len(stats['columns'])}")
        c3.metric(
            "Time Span",
            f"{stats['time_span_h']} hrs"
            if stats["time_span_h"] else "N/A"
        )
        c4.metric(
            "Max GPUs",
            f"{stats['max_gpus']:,}"
            if stats["max_gpus"] else "N/A"
        )
        st.markdown("### Dataset Columns")
        st.code(", ".join(cols), language=None)

    else:
        st.info(
            "👆 Click **Load RAPS Data from Zenodo** to fetch "
            "real HPC job traces. Until then, simulation "
            "uses synthetic data."
        )
        st.markdown("### Synthetic vs Real Data Comparison")
        compare_data = {
            "Feature": [
                "Data Source",
                "Job Patterns",
                "Power Ramps",
                "Load Drops",
                "Realism",
                "Speed"
            ],
            "🔢 Synthetic (Phase 1)": [
                "numpy random",
                "Regular intervals",
                "Fixed ramp rate",
                "Predictable",
                "Low",
                "Instant"
            ],
            "📂 Real RAPS (Phase 2)": [
                "Zenodo / Oak Ridge NL",
                "Real HPC scheduling",
                "Variable job-dependent",
                "Unpredictable",
                "High",
                "~1-2 min download"
            ]
        }
        st.table(pd.DataFrame(compare_data))

# ── About Tab ──────────────────────────────────────────────────
with tab6:
    st.subheader("ℹ️ About This Tool")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### What This Simulates")
        st.markdown(
            "This tool models the power dynamics of an AI Data "
            "Center powered by a **Gas Generator + BESS**."
        )
        st.markdown("**Phase 1:** Synthetic AI workload profiles")
        st.markdown("**Phase 2A:** Real ExaDigiT RAPS job trace data")
        st.markdown(
            "**Phase 2B (coming):** More NVIDIA tests (7, 8, 10)"
        )
        st.markdown(
            "**Phase 2C (coming):** Animated power flow diagram"
        )
        st.markdown("### Key Question Answered")
        st.markdown(
            "> When AI load suddenly drops, does the BESS "
            "protect the gas generator — or do we need "
            "dummy load banks?"
        )

    with col2:
        st.markdown("### NVIDIA Tests Implemented")
        test_data = {
            "Test": ["Test 2", "Test 4", "Test 9", "Test 11"],
            "Description": [
                "GFM Voltage & Frequency Regulation",
                "AI Buffering Proxy (20% IT load/sec)",
                "Generator Following (droop response)",
                "SOC Drift (24h energy management)"
            ],
            "Status": [
                "✅ Live", "✅ Live", "✅ Live", "✅ Live"
            ]
        }
        st.table(pd.DataFrame(test_data))

        st.markdown("### System Architecture")
        st.code(
            "Gas Generator (droop master)\n"
            "        ↕\n"
            "Grid-Forming BESS (fast buffer)\n"
            "        ↕\n"
            "AI GPU Load Racks\n"
            "        ↕\n"
            "Dummy Load Banks (backup)",
            language=None
        )

    st.divider()
    st.markdown("### Roadmap")
    roadmap = {
        "Phase": [
            "Phase 1", "Phase 2A", "Phase 2B", "Phase 2C"
        ],
        "Feature": [
            "Synthetic simulation + NVIDIA Tests 2,4,9,11",
            "Real RAPS data from Zenodo",
            "More NVIDIA Tests (7, 8, 10)",
            "Animated power flow diagram"
        ],
        "Status": [
            "✅ Complete",
            "✅ Complete",
            "🔜 Coming",
            "🔜 Coming"
        ]
    }
    st.table(pd.DataFrame(roadmap))

    st.divider()
    st.markdown("### References")
    st.markdown(
        "- NVIDIA BESS Self-Qualification Guidelines v0.4 "
        "(February 2026)"
    )
    st.markdown("- ExaDigiT RAPS — Oak Ridge National Laboratory")
    st.markdown("- EPRI DCFlex Initiative")
    st.markdown("- IEEE 2800 / IEEE 1547-2018")
    st.markdown(
        "- Zenodo Dataset DOI: 10.5281/zenodo.10127767"
    )

# ── Run Simulation ─────────────────────────────────────────────
if run_sim:
    with st.spinner("⚙️ Running simulation..."):
        try:
            gpu_map = {
                "H100 (700W)":  700,
                "A100 (400W)":  400,
                "H200 (1000W)": 1000
            }
            gpu_power_w = gpu_map[gpu_power_kw]

            use_raps = (
                "Real RAPS" in data_source
                and "raps_loaded" in st.session_state
            )

            if use_raps:
                load_model = RAPSLoader(
                    it_load_mw   = it_load_mw,
                    gpu_power_w  = gpu_power_w,
                    duration_min = sim_duration
                )
                data_label = "📂 Real RAPS Data (ExaDigiT)"
            else:
                load_model = AILoadProfile(
                    it_load_mw    = it_load_mw,
                    num_gpus      = num_gpus,
                    gpu_power_w   = gpu_power_w,
                    ramp_rate_pct = ramp_rate,
                    duration_min  = sim_duration
                )
                data_label = "🔢 Synthetic Data"

            gen_model = GasGenerator(
                rated_mw        = gen_rated_mw,
                droop_pct       = droop_pct,
                governor_time_s = governor_speed,
                min_load_pct    = min_load_pct
            )
            bess_obj = BESSModel(
                capacity_mwh    = bess_capacity,
                max_power_mw    = bess_power,
                soc_initial_pct = soc_initial,
                soc_min_pct     = soc_min,
                soc_max_pct     = soc_max,
                mode            = bess_mode,
                dummy_load_mw   = dummy_load_mw
            )

            engine  = SimulationEngine(
                load_model, gen_model, bess_obj
            )
            results = engine.run()
            df      = results["timeseries"]
            tests   = results["nvidia_tests"]

            st.success(
                f"✅ Simulation complete! "
                f"Data source: {data_label}"
            )

        except Exception as e:
            st.error(f"❌ Simulation error: {e}")
            st.stop()

    # ── Tab 1: AI Load Profile ──────────────────────────────────
    with tab1:
        st.subheader("📊 AI Workload Power Demand Profile")
        if use_raps:
            st.success("📂 Using real ExaDigiT RAPS job trace data")
        else:
            st.info("🔢 Using synthetic workload data")

        st.caption(
            "NVIDIA Test 4: Ramp rate must be "
            "<= 20% IT load/second"
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Peak Load",
                  f"{df['it_load_mw'].max():.1f} MW")
        c2.metric("Min Load",
                  f"{df['it_load_mw'].min():.1f} MW")
        c3.metric("Avg Load",
                  f"{df['it_load_mw'].mean():.1f} MW")
        c4.metric("Load Drop Events",
                  f"{int(results['num_drop_events'])}")

        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=df["time_s"],
            y=df["it_load_mw"],
            name="IT Load (MW)",
            line=dict(color="#00D4FF", width=2),
            fill="tozeroy",
            fillcolor="rgba(0,212,255,0.1)"
        ))
        for event in results["drop_events"]:
            fig1.add_vrect(
                x0=event["start"],
                x1=event["end"],
                fillcolor="rgba(255,50,50,0.2)",
                annotation_text="Load Drop",
                annotation_position="top left",
                line_width=0
            )
        fig1.update_layout(
            title=f"AI Workload Power Profile — {data_label}",
            xaxis_title="Time (seconds)",
            yaxis_title="Power (MW)",
            template="plotly_dark",
            height=420
        )
        st.plotly_chart(fig1, use_container_width=True)

        if results["drop_events"]:
            st.subheader("📋 Load Drop Event Details")
            drop_df = pd.DataFrame(results["drop_events"])
            drop_df.columns = [
                "Start (s)", "End (s)",
                "Drop (MW)", "Drop (%)", "Ramp (MW/s)"
            ]
            st.dataframe(drop_df, use_container_width=True)
            st.download_button(
                "⬇️ Download Drop Events CSV",
                drop_df.to_csv(index=False),
                "load_drop_events.csv",
                "text/csv"
            )

    # ── Tab 2: Generator Response ───────────────────────────────
    with tab2:
        st.subheader("⛽ Gas Generator Response to Load Changes")
        st.caption(
            "NVIDIA Test 9: Frequency must stay within "
            "limits during load transients"
        )

        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Max Freq Deviation",
            f"{df['freq_deviation_hz'].abs().max():.3f} Hz",
            delta="Limit: 0.5 Hz"
        )
        c2.metric(
            "Overspeed Events",
            f"{int(results['overspeed_events'])}",
            delta="Target: 0"
        )
        c3.metric(
            "Generator Utilization",
            f"{df['gen_output_mw'].mean() / gen_rated_mw * 100:.1f}%"
        )

        fig2 = make_subplots(
            rows=2, cols=1,
            subplot_titles=(
                "Generator Output (MW)",
                "Frequency Deviation (Hz)"
            )
        )
        fig2.add_trace(go.Scatter(
            x=df["time_s"], y=df["gen_output_mw"],
            name="Generator Output",
            line=dict(color="#FFB300", width=2)
        ), row=1, col=1)
        fig2.add_hline(
            y=gen_rated_mw * min_load_pct / 100,
            line_dash="dash", line_color="red",
            annotation_text="Min Load — wet stacking risk",
            row=1, col=1
        )
        fig2.add_hline(
            y=gen_rated_mw,
            line_dash="dash", line_color="orange",
            annotation_text="Rated Capacity",
            row=1, col=1
        )
        fig2.add_trace(go.Scatter(
            x=df["time_s"], y=df["freq_deviation_hz"],
            name="Frequency Deviation",
            line=dict(color="#FF5733", width=2)
        ), row=2, col=1)
        fig2.add_hline(
            y=0.5, line_dash="dash", line_color="orange",
            annotation_text="Warning +0.5 Hz", row=2, col=1
        )
        fig2.add_hline(
            y=-0.5, line_dash="dash",
            line_color="orange", row=2, col=1
        )
        fig2.add_hline(
            y=1.0, line_dash="dash", line_color="red",
            annotation_text="Critical +1.0 Hz", row=2, col=1
        )
        fig2.add_hline(
            y=-1.0, line_dash="dash",
            line_color="red", row=2, col=1
        )
        fig2.update_layout(template="plotly_dark", height=600)
        st.plotly_chart(fig2, use_container_width=True)

        st.download_button(
            "⬇️ Download Generator Data CSV",
            df[["time_s", "gen_output_mw",
                "freq_deviation_hz"]].to_csv(index=False),
            "generator_response.csv",
            "text/csv"
        )

    # ── Tab 3: BESS Behavior ────────────────────────────────────
    with tab3:
        st.subheader("🔋 BESS State of Charge & Power Flow")
        st.caption(
            "NVIDIA Test 11: SOC drift must stay "
            "<= 5% over simulation period"
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Final SOC",
                  f"{df['soc_pct'].iloc[-1]:.1f}%")
        c2.metric("SOC Drift",
                  f"{results['soc_drift']:.2f}%",
                  delta="Pass if <= 5%")
        c3.metric("Energy Absorbed",
                  f"{results['energy_absorbed_mwh']:.2f} MWh")
        c4.metric("Cost Saving",
                  f"${results['cost_saving_usd']:,.0f}")

        fig3 = make_subplots(
            rows=2, cols=1,
            subplot_titles=(
                "BESS State of Charge (%)",
                "BESS Power Flow — Positive=Charging, "
                "Negative=Discharging"
            )
        )
        fig3.add_trace(go.Scatter(
            x=df["time_s"], y=df["soc_pct"],
            name="SOC (%)",
            line=dict(color="#00FF88", width=2),
            fill="tozeroy",
            fillcolor="rgba(0,255,136,0.1)"
        ), row=1, col=1)
        fig3.add_hline(
            y=soc_max, line_dash="dash", line_color="red",
            annotation_text=f"SOC Max {soc_max}%",
            row=1, col=1
        )
        fig3.add_hline(
            y=soc_min, line_dash="dash", line_color="orange",
            annotation_text=f"SOC Min {soc_min}%",
            row=1, col=1
        )
        fig3.add_trace(go.Scatter(
            x=df["time_s"], y=df["bess_power_mw"],
            name="BESS Power (MW)",
            line=dict(color="#BB86FC", width=2)
        ), row=2, col=1)
        fig3.add_hline(
            y=0, line_color="white",
            line_dash="dot", row=2, col=1
        )
        fig3.update_layout(template="plotly_dark", height=600)
        st.plotly_chart(fig3, use_container_width=True)

        st.download_button(
            "⬇️ Download BESS Data CSV",
            df[["time_s", "soc_pct",
                "bess_power_mw"]].to_csv(index=False),
            "bess_behavior.csv",
            "text/csv"
        )

    # ── Tab 4: BESS vs Dummy Load ───────────────────────────────
    with tab4:
        st.subheader("⚖️ BESS Charging vs Dummy Load Bank")
        st.caption(
            "How much surplus power stored in BESS "
            "vs wasted as heat in dummy loads"
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Stored in BESS",
                  f"{results['energy_absorbed_mwh']:.2f} MWh",
                  delta="Saved")
        c2.metric("Wasted as Heat",
                  f"{results['dummy_load_mwh']:.2f} MWh",
                  delta="Dummy Load")
        c3.metric("Generator Reduced",
                  f"{results['gen_reduced_mwh']:.2f} MWh",
                  delta="Fuel Saved")

        col1, col2 = st.columns(2)
        with col1:
            fig_pie = go.Figure(go.Pie(
                labels=[
                    "BESS Absorbed",
                    "Dummy Load Wasted",
                    "Generator Reduced"
                ],
                values=[
                    results["energy_absorbed_mwh"],
                    results["dummy_load_mwh"],
                    results["gen_reduced_mwh"]
                ],
                marker_colors=["#00FF88", "#FF5733", "#FFB300"],
                hole=0.45
            ))
            fig_pie.update_layout(
                title="Surplus Power Distribution",
                template="plotly_dark",
                height=380
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            fig_bar = go.Figure(go.Bar(
                x=["BESS Absorbed",
                   "Dummy Load Wasted",
                   "Gen Reduced"],
                y=[
                    results["energy_absorbed_mwh"],
                    results["dummy_load_mwh"],
                    results["gen_reduced_mwh"]
                ],
                marker_color=["#00FF88", "#FF5733", "#FFB300"],
                text=[
                    f"{results['energy_absorbed_mwh']:.2f} MWh",
                    f"{results['dummy_load_mwh']:.2f} MWh",
                    f"{results['gen_reduced_mwh']:.2f} MWh"
                ],
                textposition="auto"
            ))
            fig_bar.update_layout(
                title="Energy Balance Comparison",
                yaxis_title="Energy (MWh)",
                template="plotly_dark",
                height=380
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(
            x=df["time_s"], y=df["bess_charging_mw"],
            name="BESS Charging (MW)",
            stackgroup="one",
            line=dict(color="#00FF88"),
            fillcolor="rgba(0,255,136,0.4)"
        ))
        fig4.add_trace(go.Scatter(
            x=df["time_s"], y=df["dummy_load_active_mw"],
            name="Dummy Load Active (MW)",
            stackgroup="one",
            line=dict(color="#FF5733"),
            fillcolor="rgba(255,87,51,0.4)"
        ))
        fig4.update_layout(
            title="Surplus Absorption Over Time — BESS vs Dummy Load",
            xaxis_title="Time (seconds)",
            yaxis_title="Power (MW)",
            template="plotly_dark",
            height=400
        )
        st.plotly_chart(fig4, use_container_width=True)

        st.download_button(
            "⬇️ Download Comparison CSV",
            df[["time_s", "bess_charging_mw",
                "dummy_load_active_mw"]].to_csv(index=False),
            "bess_vs_dummy.csv",
            "text/csv"
        )

    # ── Tab 5: NVIDIA Test Results ──────────────────────────────
    with tab5:
        st.subheader("✅ NVIDIA BESS Qualification Test Results")
        st.caption(
            "Based on NVIDIA BESS Self-Qualification "
            "Guidelines v0.4 — February 2026"
        )

        pass_count = sum(
            1 for t in tests.values() if t["status"] == "PASS"
        )
        total = len(tests)

        c1, c2, c3 = st.columns(3)
        c1.metric("Tests Passed", f"{pass_count} / {total}")
        c2.metric(
            "Overall Status",
            "QUALIFIED" if pass_count == total
            else "NOT QUALIFIED"
        )
        c3.metric(
            "Compliance Score",
            f"{pass_count / total * 100:.0f}%"
        )

        st.divider()

        for test_name, test_result in tests.items():
            status = test_result["status"]
            icon   = "✅" if status == "PASS" else "❌"

            with st.expander(
                f"{icon} {test_name} — {status}",
                expanded=(status == "FAIL")
            ):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Required Criteria:**")
                    for k, v in test_result["criteria"].items():
                        st.write(f"- {k}: `{v}`")
                with col2:
                    st.markdown("**Measured Values:**")
                    for k, v in test_result["measured"].items():
                        ok   = test_result.get(
                            "details", {}
                        ).get(k, True)
                        tick = "✅" if ok else "❌"
                        st.write(f"{tick} {k}: `{v}`")

                rec = test_result.get("recommendation")
                if rec and str(rec) != "None":
                    st.warning(f"💡 Recommendation: {rec}")

        st.divider()
        summary_rows = []
        for test_name, test_result in tests.items():
            for k, v in test_result["measured"].items():
                summary_rows.append({
                    "Test":      test_name,
                    "Parameter": k,
                    "Measured":  v,
                    "Status":    test_result["status"]
                })
        st.download_button(
            "⬇️ Download Full Test Results CSV",
            pd.DataFrame(summary_rows).to_csv(index=False),
            "nvidia_test_results.csv",
            "text/csv"
        )

else:
    with tab1:
        st.info(
            "👈 Configure parameters in the sidebar "
            "and click RUN SIMULATION"
        )
        st.markdown("### How to Use")
        st.markdown(
            "1. Choose **Data Source** in sidebar "
            "(Synthetic or Real RAPS)"
        )
        st.markdown("2. Set **AI Data Center** parameters")
        st.markdown("3. Configure **Gas Generator** settings")
        st.markdown("4. Set **BESS** parameters")
        st.markdown("5. Set **Dummy Load Bank** capacity")
        st.markdown("6. Click **RUN SIMULATION**")
        st.markdown(
            "7. Explore all tabs and download CSV results"
        )
    with tab2:
        st.info(
            "👈 Run simulation to see generator frequency response"
        )
    with tab3:
        st.info(
            "👈 Run simulation to see BESS SOC and power flow"
        )
    with tab4:
        st.info(
            "👈 Run simulation to see "
            "BESS vs Dummy Load comparison"
        )
    with tab5:
        st.info(
            "👈 Run simulation to see "
            "NVIDIA qualification test results"
        )
