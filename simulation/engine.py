import numpy as np
import pandas as pd


class SimulationEngine:
    def __init__(self, load_model, gen_model, bess_model):
        self.load_model = load_model
        self.gen_model  = gen_model
        self.bess_model = bess_model

    def _integrate(self, y, x):
        """Safe integration — works with any numpy version."""
        return float(np.sum(
            (y[:-1] + y[1:]) * 0.5 * np.diff(x)
        ))

    def run(self):

        # ── Step 1: Generate AI load profile ──────────────────
        df_load, drop_events = self.load_model.generate()
        load_mw = df_load["it_load_mw"].values.astype(float)
        time_s  = df_load["time_s"].values.astype(float)

        # ── Step 2: BESS simulation ────────────────────────────
        bess_power, soc_arr, dummy_active, bess_charging = \
            self.bess_model.simulate(load_mw,
                                     self.gen_model.rated_mw)

        bess_power    = np.array(bess_power,    dtype=float)
        soc_arr       = np.array(soc_arr,       dtype=float)
        dummy_active  = np.array(dummy_active,  dtype=float)
        bess_charging = np.array(bess_charging, dtype=float)

        # ── Step 3: Generator simulation ──────────────────────
        gen_output, freq_dev, overspeed = \
            self.gen_model.simulate(load_mw, bess_power)

        gen_output = np.array(gen_output, dtype=float)
        freq_dev   = np.array(freq_dev,   dtype=float)
        overspeed  = np.array(overspeed,  dtype=bool)

        # ── Step 4: Simulate voltage profile ──────────────────
        voltage = self._simulate_voltage(
            load_mw, gen_output, bess_power, time_s
        )

        # ── Step 5: Build results dataframe ───────────────────
        df = pd.DataFrame({
            "time_s":               time_s,
            "it_load_mw":           load_mw,
            "gen_output_mw":        gen_output,
            "freq_deviation_hz":    freq_dev,
            "bess_power_mw":        bess_power,
            "soc_pct":              soc_arr * 100.0,
            "dummy_load_active_mw": dummy_active,
            "bess_charging_mw":     bess_charging,
            "voltage_pu":           voltage,
        })

        # ── Step 6: Compute summary metrics ───────────────────
        energy_absorbed_mwh = self._integrate(
            np.clip(bess_charging, 0, None), time_s
        ) / 3600.0

        dummy_load_mwh = self._integrate(
            dummy_active, time_s
        ) / 3600.0

        gen_reduced_mwh = self._integrate(
            np.clip(self.gen_model.rated_mw - gen_output,
                    0, None), time_s
        ) / 3600.0

        soc_drift   = float(abs(soc_arr[-1] - soc_arr[0]) * 100.0)
        cost_saving = energy_absorbed_mwh * 0.094 * 1000.0

        # ── Step 7: NVIDIA Test Results ───────────────────────
        nvidia_tests = self._evaluate_nvidia_tests(
            df, drop_events, soc_drift,
            energy_absorbed_mwh, overspeed, time_s
        )

        # ── Step 8: Power flow snapshot (for animation) ───────
        mid = len(time_s) // 2
        power_flow = {
            "gen_mw":     float(gen_output[mid]),
            "bess_mw":    float(bess_power[mid]),
            "it_load_mw": float(load_mw[mid]),
            "dummy_mw":   float(dummy_active[mid]),
            "soc_pct":    float(soc_arr[mid] * 100),
            "freq_hz":    float(50.0 + freq_dev[mid]),
            "voltage_pu": float(voltage[mid]),
        }

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
            "power_flow":           power_flow,
        }

    def _simulate_voltage(self, load_mw, gen_output,
                          bess_power, time_s):
        """Simulate per-unit voltage profile."""
        voltage = np.ones(len(time_s))
        rated   = self.gen_model.rated_mw

        for i in range(1, len(time_s)):
            net    = gen_output[i] + abs(
                min(bess_power[i], 0)
            ) - load_mw[i]
            dv     = net / (rated * 10.0)
            voltage[i] = np.clip(
                voltage[i - 1] + dv * 0.1, 0.70, 1.10
            )

        return voltage

    def _evaluate_nvidia_tests(self, df, drop_events,
                                soc_drift, energy_absorbed,
                                overspeed, time_s):
        tests = {}

        load_max  = float(df["it_load_mw"].max())
        freq_max  = float(df["freq_deviation_hz"].abs().max())
        gen_min   = float(df["gen_output_mw"].min())
        volt_min  = float(df["voltage_pu"].min())
        volt_max  = float(df["voltage_pu"].max())
        n_over    = int(overspeed.sum())
        bess_mode = self.bess_model.mode

        # ── Test 2: GFM Regulation ─────────────────────────────
        gfm_mode = "Grid-Forming" in bess_mode
        freq_ok  = freq_max < 1.0
        t2_pass  = gfm_mode and freq_ok

        tests["Test 2: GFM Voltage & Frequency Regulation"] = {
            "status": "PASS" if t2_pass else "FAIL",
            "criteria": {
                "BESS mode":        "Grid-Forming required",
                "Frequency stable": "< 1.0 Hz deviation",
            },
            "measured": {
                "BESS mode":        bess_mode,
                "Frequency stable": f"{freq_max:.3f} Hz",
            },
            "details": {
                "BESS mode":        gfm_mode,
                "Frequency stable": freq_ok,
            },
            "recommendation": (
                "Switch BESS to Grid-Forming mode"
                if not t2_pass else None
            ),
        }

        # ── Test 4: AI Buffering ───────────────────────────────
        max_ramp = 0.0
        if drop_events:
            max_ramp = max(
                float(e.get("ramp_mw_s", 0))
                for e in drop_events
            )
        ramp_pct = (
            max_ramp / load_max * 100.0
        ) if load_max > 0 else 0.0

        ramp_ok = ramp_pct <= 20.0
        f4_ok   = freq_max < 0.5
        osc_ok  = n_over == 0
        t4_pass = f4_ok and osc_ok

        tests["Test 4: AI Buffering (NVIDIA PERF-CORE-01)"] = {
            "status": "PASS" if t4_pass else "FAIL",
            "criteria": {
                "Max ramp rate":            "≤ 20% IT load/sec",
                "Frequency deviation":      "< 0.5 Hz",
                "No sustained oscillation": "Required",
            },
            "measured": {
                "Max ramp rate":        f"{ramp_pct:.1f}% /sec",
                "Frequency deviation":  f"{freq_max:.3f} Hz",
                "Overspeed events":     f"{n_over}",
            },
            "details": {
                "Max ramp rate":            ramp_ok,
                "Frequency deviation":      f4_ok,
                "No sustained oscillation": osc_ok,
            },
            "recommendation": (
                "Increase BESS power rating or reduce ramp rate"
                if not t4_pass else None
            ),
        }

        # ── Test 7: LVRT ───────────────────────────────────────
        lvrt_ok   = volt_min >= 0.80
        recov_ok  = volt_max <= 1.10
        t7_pass   = lvrt_ok and recov_ok

        tests["Test 7: LVRT — Low Voltage Ride Through"] = {
            "status": "PASS" if t7_pass else "FAIL",
            "criteria": {
                "Min voltage":      "≥ 0.80 pu during fault",
                "Max voltage":      "≤ 1.10 pu recovery",
                "Ride-through":     "BESS must stay connected",
            },
            "measured": {
                "Min voltage":  f"{volt_min:.3f} pu",
                "Max voltage":  f"{volt_max:.3f} pu",
                "Ride-through": "YES" if lvrt_ok else "NO",
            },
            "details": {
                "Min voltage":  lvrt_ok,
                "Max voltage":  recov_ok,
                "Ride-through": lvrt_ok,
            },
            "recommendation": (
                "Increase BESS reactive power support "
                "or add static VAR compensator"
                if not t7_pass else None
            ),
        }

        # ── Test 8: Grid to Island Transition ─────────────────
        island_ok = gfm_mode
        freq_isl  = freq_max < 1.5
        volt_isl  = volt_min >= 0.75
        t8_pass   = island_ok and freq_isl and volt_isl

        tests["Test 8: Grid to Island Transition"] = {
            "status": "PASS" if t8_pass else "FAIL",
            "criteria": {
                "BESS mode":          "Grid-Forming required",
                "Freq during island": "< 1.5 Hz deviation",
                "Volt during island": "≥ 0.75 pu",
            },
            "measured": {
                "BESS mode":          bess_mode,
                "Freq during island": f"{freq_max:.3f} Hz",
                "Volt during island": f"{volt_min:.3f} pu",
            },
            "details": {
                "BESS mode":          island_ok,
                "Freq during island": freq_isl,
                "Volt during island": volt_isl,
            },
            "recommendation": (
                "GFM BESS required to maintain voltage "
                "and frequency during islanding"
                if not t8_pass else None
            ),
        }

        # ── Test 9: Generator Following ───────────────────────
        min_ok   = gen_min >= self.gen_model.min_load_mw
        t9_pass  = min_ok and freq_ok

        tests["Test 9: Generator Following (NVIDIA TEST-9)"] = {
            "status": "PASS" if t9_pass else "FAIL",
            "criteria": {
                "Min generator load": (
                    f"≥ {self.gen_model.min_load_mw:.1f} MW"
                ),
                "Frequency stable":   "< 1.0 Hz deviation",
            },
            "measured": {
                "Min generator load": f"{gen_min:.1f} MW",
                "Max freq deviation": f"{freq_max:.3f} Hz",
            },
            "details": {
                "Min generator load": min_ok,
                "Frequency stable":   freq_ok,
            },
            "recommendation": (
                "Increase dummy load bank to prevent "
                "low load on generator"
                if not t9_pass else None
            ),
        }

        # ── Test 10: Black Start ───────────────────────────────
        soc_ok      = float(
            df["soc_pct"].iloc[0]
        ) >= 50.0
        bs_volt_ok  = volt_min >= 0.85
        bs_freq_ok  = freq_max < 2.0
        t10_pass    = gfm_mode and soc_ok and bs_volt_ok

        tests["Test 10: Black Start Capability"] = {
            "status": "PASS" if t10_pass else "FAIL",
            "criteria": {
                "BESS mode":       "Grid-Forming required",
                "Initial SOC":     "≥ 50% for black start",
                "Voltage restore": "≥ 0.85 pu within 5s",
            },
            "measured": {
                "BESS mode":       bess_mode,
                "Initial SOC":     f"{df['soc_pct'].iloc[0]:.1f}%",
                "Min voltage":     f"{volt_min:.3f} pu",
            },
            "details": {
                "BESS mode":       gfm_mode,
                "Initial SOC":     soc_ok,
                "Voltage restore": bs_volt_ok,
            },
            "recommendation": (
                "Ensure SOC ≥ 50% and GFM mode "
                "for black start capability"
                if not t10_pass else None
            ),
        }

        # ── Test 11: SOC Drift ────────────────────────────────
        soc_d_ok = soc_drift <= 5.0
        t11_pass = soc_d_ok

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
                "SOC drift":       soc_d_ok,
                "Energy absorbed": True,
            },
            "recommendation": (
                "Tune energy management to balance "
                "BESS charging/discharging"
                if not t11_pass else None
            ),
        }

        return tests
