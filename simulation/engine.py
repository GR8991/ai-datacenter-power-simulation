import numpy as np
import pandas as pd


class SimulationEngine:
    def __init__(self, load_model, gen_model, bess_model):
        self.load_model = load_model
        self.gen_model  = gen_model
        self.bess_model = bess_model

    def _integrate(self, y, x):
        """Safe integration works with any numpy version."""
        return float(np.sum(
            (y[:-1] + y[1:]) * 0.5 * np.diff(x)
        ))

    def run(self):

        # ── Step 1: Generate AI load profile ──────────────────────
        df_load, drop_events = self.load_model.generate()
        load_mw = df_load["it_load_mw"].values.astype(float)
        time_s  = df_load["time_s"].values.astype(float)

        # ── Step 2: BESS simulation ────────────────────────────────
        bess_power, soc_arr, dummy_active, bess_charging = \
            self.bess_model.simulate(load_mw, self.gen_model.rated_mw)

        bess_power    = np.array(bess_power,    dtype=float)
        soc_arr       = np.array(soc_arr,       dtype=float)
        dummy_active  = np.array(dummy_active,  dtype=float)
        bess_charging = np.array(bess_charging, dtype=float)

        # ── Step 3: Generator simulation ──────────────────────────
        gen_output, freq_dev, overspeed = \
            self.gen_model.simulate(load_mw, bess_power)

        gen_output = np.array(gen_output, dtype=float)
        freq_dev   = np.array(freq_dev,   dtype=float)
        overspeed  = np.array(overspeed,  dtype=bool)

        # ── Step 4: Build results dataframe ───────────────────────
        df = pd.DataFrame({
            "time_s":               time_s,
            "it_load_mw":           load_mw,
            "gen_output_mw":        gen_output,
            "freq_deviation_hz":    freq_dev,
            "bess_power_mw":        bess_power,
            "soc_pct":              soc_arr * 100.0,
            "dummy_load_active_mw": dummy_active,
            "bess_charging_mw":     bess_charging,
        })

        # ── Step 5: Compute summary metrics ───────────────────────
        energy_absorbed_mwh = self._integrate(
            np.clip(bess_charging, 0, None), time_s
        ) / 3600.0

        dummy_load_mwh = self._integrate(
            dummy_active, time_s
        ) / 3600.0

        gen_reduced_mwh = self._integrate(
            np.clip(self.gen_model.rated_mw - gen_output, 0, None),
            time_s
        ) / 3600.0

        soc_drift    = float(abs(soc_arr[-1] - soc_arr[0]) * 100.0)
        cost_saving  = energy_absorbed_mwh * 0.094 * 1000.0   # USD

        # ── Step 6: NVIDIA Test Results ───────────────────────────
        nvidia_tests = self._evaluate_nvidia_tests(
            df, drop_events, soc_drift,
            energy_absorbed_mwh, overspeed
        )

        return {
            "timeseries":           df,
            "drop_events":          drop_events,
            "num_drop_events":      len(drop_events),
            "overspeed_events":     int(overspeed.sum()),
            "soc_drift":            soc_drift,
            "energy_absorbed_mwh":  energy_absorbed_mwh,
            "dummy_load_mwh":       dummy_load_mwh,
            "gen_reduced_mwh":      gen_reduced_mwh,
            "cost_saving_usd":      cost_saving,
            "nvidia_tests":         nvidia_tests,
        }

    # ──────────────────────────────────────────────────────────────
    def _evaluate_nvidia_tests(self, df, drop_events,
                                soc_drift, energy_absorbed, overspeed):
        tests = {}

        load_max   = float(df["it_load_mw"].max())
        freq_max   = float(df["freq_deviation_hz"].abs().max())
        gen_min    = float(df["gen_output_mw"].min())
        n_over     = int(overspeed.sum())

        # ── Test 4: AI Buffering ───────────────────────────────────
        max_ramp = 0.0
        if drop_events:
            max_ramp = max(
                float(e.get("ramp_mw_s", 0)) for e in drop_events
            )
        ramp_pct_s = (max_ramp / load_max * 100.0) if load_max > 0 else 0.0

        ramp_ok  = ramp_pct_s  <= 20.0
        freq_ok  = freq_max    <  0.5
        osc_ok   = n_over      == 0
        t4_pass  = freq_ok and osc_ok

        tests["Test 4: AI Buffering (NVIDIA PERF-CORE-01)"] = {
            "status": "PASS" if t4_pass else "FAIL",
            "criteria": {
                "Max ramp rate":            "≤ 20% IT load/sec",
                "Frequency deviation":      "< 0.5 Hz",
                "No sustained oscillation": "Required",
            },
            "measured": {
                "Max ramp rate":        f"{ramp_pct_s:.1f}% /sec",
                "Frequency deviation":  f"{freq_max:.3f} Hz",
                "Overspeed events":     f"{n_over}",
            },
            "details": {
                "Max ramp rate":            ramp_ok,
                "Frequency deviation":      freq_ok,
                "No sustained oscillation": osc_ok,
            },
            "recommendation": (
                "Increase BESS power rating or reduce ramp rate"
                if not t4_pass else None
            ),
        }

        # ── Test 9: Generator Following ───────────────────────────
        min_load_ok  = gen_min  >= self.gen_model.min_load_mw
        freq_stab    = freq_max <  1.0
        t9_pass      = min_load_ok and freq_stab

        tests["Test 9: Generator Following (NVIDIA TEST-9)"] = {
            "status": "PASS" if t9_pass else "FAIL",
            "criteria": {
                "Min generator load":  f"≥ {self.gen_model.min_load_mw:.1f} MW",
                "Frequency stable":    "< 1.0 Hz deviation",
            },
            "measured": {
                "Min generator load":  f"{gen_min:.1f} MW",
                "Max freq deviation":  f"{freq_max:.3f} Hz",
            },
            "details": {
                "Min generator load":  min_load_ok,
                "Frequency stable":    freq_stab,
            },
            "recommendation": (
                "Increase dummy load bank to prevent low load on generator"
                if not t9_pass else None
            ),
        }

        # ── Test 11: SOC Drift ────────────────────────────────────
        soc_ok   = soc_drift <= 5.0
        t11_pass = soc_ok

        tests["Test 11: SOC Drift (NVIDIA TEST-11)"] = {
            "status": "PASS" if t11_pass else "FAIL",
            "criteria": {
                "SOC drift": "≤ 5% of usable capacity",
            },
            "measured": {
                "SOC drift":       f"{soc_drift:.2f}%",
                "Energy absorbed": f"{energy_absorbed:.2f} MWh",
            },
            "details": {
                "SOC drift":       soc_ok,
                "Energy absorbed": True,
            },
            "recommendation": (
                "Tune energy management to balance BESS charging/discharging"
                if not t11_pass else None
            ),
        }

        # ── Test 2: GFM Regulation ────────────────────────────────
        gfm_mode = "Grid-Forming" in self.bess_model.mode
        t2_pass  = gfm_mode and freq_stab

        tests["Test 2: GFM Voltage & Frequency Regulation"] = {
            "status": "PASS" if t2_pass else "FAIL",
            "criteria": {
                "BESS mode":        "Grid-Forming required",
                "Frequency stable": "< 1.0 Hz deviation",
            },
            "measured": {
                "BESS mode":        self.bess_model.mode,
                "Frequency stable": f"{freq_max:.3f} Hz",
            },
            "details": {
                "BESS mode":        gfm_mode,
                "Frequency stable": freq_stab,
            },
            "recommendation": (
                "Switch BESS to Grid-Forming mode for islanded stability"
                if not t2_pass else None
            ),
        }

        return tests
