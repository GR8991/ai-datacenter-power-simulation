import pandas as pd
import numpy as np
import os


# Module-level cache
_cache = {}


class RAPSLoader:
    """
    Loads ExaDigiT RAPS-style job trace data.
    Uses a lightweight sample CSV from the repo —
    no large file download needed.
    """

    # Path to sample data in the repo
    SAMPLE_CSV = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "sample_jobs.csv"
    )

    ZENODO_URL = (
        "https://zenodo.org/records/10127767"
        "/files/job_table.parquet?download=1"
    )

    def __init__(self,
                 it_load_mw,
                 gpu_power_w  = 700,
                 duration_min = 30,
                 parquet_path = None):
        self.it_load_mw   = it_load_mw
        self.gpu_power_w  = gpu_power_w
        self.duration_s   = duration_min * 60
        self.dt           = 1
        self.parquet_path = parquet_path

    @staticmethod
    def clear_cache():
        """Clear module-level cache."""
        _cache.clear()

    def _load_sample(self):
        """Load sample CSV from repo."""
        if "sample_df" in _cache:
            return _cache["sample_df"]
        df = pd.read_csv(self.SAMPLE_CSV)
        _cache["sample_df"] = df
        return df

    def load(self):
        """Load job data — from sample CSV."""
        return self._load_sample()

    def get_columns(self):
        """Return available columns."""
        return list(self.load().columns)

    def get_job_stats(self):
        """Return summary statistics."""
        df    = self.load()
        stats = {
            "total_jobs":  len(df),
            "columns":     list(df.columns),
            "time_span_h": None,
            "max_gpus":    None,
            "data_source": "Sample Dataset (RAPS-compatible format)",
            "note": (
                "Sample of 100 jobs in RAPS format. "
                "Full dataset: zenodo.org/records/10127767"
            )
        }
        if "start_time" in df.columns and "end_time" in df.columns:
            span = df["end_time"].max() - df["start_time"].min()
            stats["time_span_h"] = round(float(span) / 3600, 1)

        if "num_gpus" in df.columns:
            stats["max_gpus"] = int(df["num_gpus"].max())

        return stats

    def generate(self):
        """
        Process job table into power time series.
        Returns same format as AILoadProfile.generate()
        """
        df_jobs = self.load().copy()

        # ── Convert times to float seconds ────────────────────
        df_jobs["_start_s"] = pd.to_numeric(
            df_jobs["start_time"], errors="coerce"
        ).fillna(0)
        df_jobs["_end_s"] = pd.to_numeric(
            df_jobs["end_time"], errors="coerce"
        ).fillna(0)
        df_jobs["_gpus"] = pd.to_numeric(
            df_jobs["num_gpus"], errors="coerce"
        ).fillna(0)

        # Normalize to start from 0
        df_jobs["_start_s"] -= df_jobs["_start_s"].min()
        df_jobs["_end_s"]   -= df_jobs["_end_s"].min()

        # ── Build time array ───────────────────────────────────
        time_s = np.arange(0, self.duration_s, self.dt)
        power  = np.zeros(len(time_s))

        # ── Scale factor ───────────────────────────────────────
        max_gpus = float(df_jobs["_gpus"].max())
        max_mw   = (max_gpus * self.gpu_power_w) / 1e6
        scale    = self.it_load_mw / max(max_mw, 0.001)

        # ── Fill power for each active job ─────────────────────
        for _, job in df_jobs.iterrows():
            start = float(job["_start_s"])
            end   = float(job["_end_s"])
            gpus  = float(job["_gpus"])

            if end <= start or gpus <= 0:
                continue

            t_start = max(0, int(start))
            t_end   = min(len(time_s) - 1, int(end))

            if t_start >= len(time_s):
                continue

            job_mw = (gpus * self.gpu_power_w) / 1e6 * scale

            for t in range(t_start, t_end + 1):
                if t < len(power):
                    power[t] += job_mw

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
