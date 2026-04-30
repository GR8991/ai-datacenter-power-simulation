import pandas as pd
import numpy as np
import os


class RAPSLoader:
    """
    Loads and processes real ExaDigiT RAPS job trace data
    from job_table.parquet and converts it into a power
    demand time series for simulation.
    """

    def __init__(self, parquet_path, it_load_mw,
                 gpu_power_w=700, duration_min=30):
        self.parquet_path = parquet_path
        self.it_load_mw   = it_load_mw
        self.gpu_power_w  = gpu_power_w
        self.duration_s   = duration_min * 60
        self.dt           = 1

    def load(self):
        """Load and validate the parquet file."""
        if not os.path.exists(self.parquet_path):
            raise FileNotFoundError(
                f"RAPS data not found at {self.parquet_path}. "
                "Please upload job_table.parquet to the data/ folder."
            )
        df = pd.read_parquet(self.parquet_path)
        return df

    def get_columns(self):
        """Return available columns in the dataset."""
        df = self.load()
        return list(df.columns)

    def generate(self):
        """
        Process RAPS job table into power time series.
        Returns same format as AILoadProfile.generate()
        so it can be used as a drop-in replacement.
        """
        df_jobs = self.load()

        # Normalize time to start from 0
        if "start_time" in df_jobs.columns:
            df_jobs = df_jobs.copy()
            df_jobs["start_time"] = (
                df_jobs["start_time"] - df_jobs["start_time"].min()
            )
            if "end_time" in df_jobs.columns:
                df_jobs["end_time"] = (
                    df_jobs["end_time"] - df_jobs["end_time"].min()
                )

        # Build time array
        time_s = np.arange(0, self.duration_s, self.dt)
        power  = np.zeros(len(time_s))

        # Fill power at each second based on active jobs
        for _, job in df_jobs.iterrows():
            start = self._get_col(job, ["start_time", "start", "submit_time"], 0)
            end   = self._get_col(job, ["end_time",   "end",   "finish_time"],  0)
            gpus  = self._get_col(job, ["num_gpus",   "gpus",  "nGPUs"],        0)

            start = float(start)
            end   = float(end)
            gpus  = float(gpus)

            if end <= start or gpus <= 0:
                continue

            # Clip to simulation window
            t_start = max(0, int(start))
            t_end   = min(len(time_s) - 1, int(end))

            if t_start >= len(time_s):
                continue

            # Power contribution from this job
            job_power_mw = (gpus * self.gpu_power_w) / 1e6

            # Scale to match configured IT load
            scale = self.it_load_mw / max(
                (df_jobs.get("num_gpus",
                 df_jobs.get("gpus",
                 df_jobs.get("nGPUs",
                 pd.Series([1])))).max() * self.gpu_power_w / 1e6),
                0.001
            )
            job_power_mw *= scale

            for t in range(t_start, t_end + 1):
                if t < len(power):
                    power[t] += job_power_mw

        # Clip to IT load max
        power = np.clip(power, 0, self.it_load_mw)

        # Smooth with rolling average (5s window)
        power_series = pd.Series(power)
        power        = power_series.rolling(
            window=5, min_periods=1
        ).mean().values

        # Detect drop events
        drop_events = []
        for i in range(1, len(power)):
            prev = power[i - 1]
            curr = power[i]
            drop = prev - curr
            if (drop > self.it_load_mw * 0.1
                    and prev > self.it_load_mw * 0.2):
                ramp = drop / 1.0  # per second
                drop_events.append({
                    "start":     float(time_s[i]),
                    "end":       float(time_s[i]) + 5.0,
                    "drop_mw":   round(float(drop), 2),
                    "drop_pct":  round(float(drop / self.it_load_mw * 100), 1),
                    "ramp_mw_s": round(float(ramp), 2)
                })

        df_out = pd.DataFrame({
            "time_s":    time_s,
            "it_load_mw": power
        })

        return df_out, drop_events

    def get_job_stats(self):
        """Return summary statistics about the job dataset."""
        df = self.load()
        stats = {
            "total_jobs":  len(df),
            "columns":     list(df.columns),
            "time_span_h": None,
            "max_gpus":    None,
        }
        if "start_time" in df.columns and "end_time" in df.columns:
            span = df["end_time"].max() - df["start_time"].min()
            stats["time_span_h"] = round(float(span) / 3600, 1)
        for col in ["num_gpus", "gpus", "nGPUs"]:
            if col in df.columns:
                stats["max_gpus"] = int(df[col].max())
                break
        return stats

    @staticmethod
    def _get_col(row, candidates, default):
        """Try multiple column name candidates."""
        for col in candidates:
            if col in row.index:
                val = row[col]
                if pd.notna(val):
                    return val
        return default
