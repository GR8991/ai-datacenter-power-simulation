import pandas as pd
import numpy as np
import os
import tempfile
import requests
import streamlit as st

ZENODO_URL = (
    "https://zenodo.org/records/10127767/files/job_table.parquet"
)


class RAPSLoader:
    """
    Loads and processes real ExaDigiT RAPS job trace data
    directly from Zenodo URL — no file upload needed.
    """

    def __init__(self, it_load_mw,
                 gpu_power_w=700,
                 duration_min=30,
                 parquet_path=None):
        self.it_load_mw   = it_load_mw
        self.gpu_power_w  = gpu_power_w
        self.duration_s   = duration_min * 60
        self.dt           = 1
        self.parquet_path = parquet_path

    @staticmethod
    @st.cache_data(show_spinner="📥 Downloading RAPS data from Zenodo...")
    def _download_data():
        """
        Download job_table.parquet from Zenodo.
        Cached so it only downloads once per session.
        """
        try:
            response = requests.get(ZENODO_URL, stream=True, timeout=120)
            response.raise_for_status()

            tmp = tempfile.NamedTemporaryFile(
                delete=False, suffix=".parquet"
            )
            total    = 0
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    tmp.write(chunk)
                    total += len(chunk)
            tmp.close()
            return tmp.name, total

        except Exception as e:
            raise RuntimeError(
                f"Failed to download RAPS data: {e}. "
                "Check your internet connection."
            )

    def _get_path(self):
        """Return path to parquet — from cache or download."""
        if self.parquet_path and os.path.exists(self.parquet_path):
            return self.parquet_path
        path, _ = self._download_data()
        return path

    def load(self):
        """Load and return the raw job dataframe."""
        path = self._get_path()
        df   = pd.read_parquet(path)
        return df

    def get_columns(self):
        """Return available columns in the dataset."""
        return list(self.load().columns)

    def get_job_stats(self):
        """Return summary statistics about the dataset."""
        df    = self.load()
        stats = {
            "total_jobs": len(df),
            "columns":    list(df.columns),
            "time_span_h": None,
            "max_gpus":    None,
        }
        if "start_time" in df.columns and "end_time" in df.columns:
            span = (
                df["end_time"].max() - df["start_time"].min()
            )
            stats["time_span_h"] = round(float(span) / 3600, 1)

        for col in ["num_gpus", "gpus", "nGPUs"]:
            if col in df.columns:
                stats["max_gpus"] = int(df[col].max())
                break

        return stats

    def generate(self):
        """
        Process RAPS job table into a power time series.
        Returns same format as AILoadProfile.generate()
        so it is a drop-in replacement.
        """
        df_jobs = self.load()

        # ── Normalize timestamps to start from 0 ───────────────
        for start_col in ["start_time", "start", "submit_time"]:
            if start_col in df_jobs.columns:
                df_jobs = df_jobs.copy()
                min_t   = df_jobs[start_col].min()
                df_jobs[start_col] = df_jobs[start_col] - min_t
                break

        for end_col in ["end_time", "end", "finish_time"]:
            if end_col in df_jobs.columns:
                min_t = df_jobs[end_col].min()
                df_jobs[end_col] = df_jobs[end_col] - min_t
                break

        # ── Build time array ───────────────────────────────────
        time_s = np.arange(0, self.duration_s, self.dt)
        power  = np.zeros(len(time_s))

        # ── Get max GPU count for scaling ──────────────────────
        max_gpus = 1
        for col in ["num_gpus", "gpus", "nGPUs"]:
            if col in df_jobs.columns:
                max_gpus = max(float(df_jobs[col].max()), 1)
                break

        max_possible_mw = (max_gpus * self.gpu_power_w) / 1e6
        scale = self.it_load_mw / max(max_possible_mw, 0.001)

        # ── Fill power for each active job ─────────────────────
        for _, job in df_jobs.iterrows():
            start = self._get_col(
                job, ["start_time", "start", "submit_time"], 0
            )
            end   = self._get_col(
                job, ["end_time", "end", "finish_time"], 0
            )
            gpus  = self._get_col(
                job, ["num_gpus", "gpus", "nGPUs"], 0
            )

            start = float(start)
            end   = float(end)
            gpus  = float(gpus)

            if end <= start or gpus <= 0:
                continue

            t_start = max(0, int(start))
            t_end   = min(len(time_s) - 1, int(end))

            if t_start >= len(time_s):
                continue

            job_power_mw = (gpus * self.gpu_power_w) / 1e6 * scale

            for t in range(t_start, t_end + 1):
                if t < len(power):
                    power[t] += job_power_mw

        # ── Clip and smooth ────────────────────────────────────
        power = np.clip(power, 0, self.it_load_mw)
        power = (
            pd.Series(power)
            .rolling(window=5, min_periods=1)
            .mean()
            .values
        )

        # ── Detect drop events ─────────────────────────────────
        drop_events = []
        for i in range(1, len(power)):
            prev = power[i - 1]
            curr = power[i]
            drop = prev - curr
            if (drop > self.it_load_mw * 0.1
                    and prev > self.it_load_mw * 0.2):
                drop_events.append({
                    "start":     float(time_s[i]),
                    "end":       float(time_s[i]) + 5.0,
                    "drop_mw":   round(float(drop), 2),
                    "drop_pct":  round(
                        float(drop / self.it_load_mw * 100), 1
                    ),
                    "ramp_mw_s": round(float(drop), 2)
                })

        df_out = pd.DataFrame({
            "time_s":     time_s,
            "it_load_mw": power
        })

        return df_out, drop_events

    @staticmethod
    def _get_col(row, candidates, default):
        """Try multiple column name candidates."""
        for col in candidates:
            if col in row.index:
                val = row[col]
                if pd.notna(val):
                    return val
        return default
