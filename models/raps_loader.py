import pandas as pd
import numpy as np
import os
import tempfile


ZENODO_URL = (
    "https://zenodo.org/records/10127767/files/job_table.parquet"
)

# Module-level cache — persists across Streamlit reruns
_cache = {}


class RAPSLoader:
    """
    Loads and processes real ExaDigiT RAPS job trace data
    directly from Zenodo URL — no file upload needed.
    """

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
    def _download_data():
        """
        Download job_table.parquet from Zenodo.
        Uses module-level dict as cache.
        """
        import requests

        if "parquet_path" in _cache:
            return _cache["parquet_path"], _cache["file_size"]

        try:
            response = requests.get(
                ZENODO_URL, stream=True, timeout=180
            )
            response.raise_for_status()

            tmp = tempfile.NamedTemporaryFile(
                delete=False, suffix=".parquet"
            )
            for chunk in response.iter_content(
                chunk_size=1024 * 1024
            ):
                if chunk:
                    tmp.write(chunk)
            tmp.close()

            file_size = os.path.getsize(tmp.name)
            _cache["parquet_path"] = tmp.name
            _cache["file_size"]    = file_size

            return tmp.name, file_size

        except Exception as e:
            raise RuntimeError(
                f"Failed to download RAPS data from Zenodo: {e}"
            )

    @staticmethod
    def clear_cache():
        """Clear the module-level cache."""
        _cache.clear()

    def _get_path(self):
        """Return path to parquet — use provided or download."""
        if (self.parquet_path
                and os.path.exists(self.parquet_path)):
            return self.parquet_path
        path, _ = self._download_data()
        return path

    @staticmethod
    def _to_seconds(series):
        """
        Convert a pandas Series to float seconds.
        Handles: Timedelta, datetime, int, float.
        """
        if pd.api.types.is_timedelta64_dtype(series):
            return series.dt.total_seconds()
        elif pd.api.types.is_datetime64_any_dtype(series):
            min_t = series.min()
            return (series - min_t).dt.total_seconds()
        else:
            return pd.to_numeric(
                series, errors="coerce"
            ).fillna(0)

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
            "total_jobs":  len(df),
            "columns":     list(df.columns),
            "time_span_h": None,
            "max_gpus":    None,
        }

        start_col = next(
            (c for c in ["start_time", "start", "submit_time"]
             if c in df.columns), None
        )
        end_col = next(
            (c for c in ["end_time", "end", "finish_time"]
             if c in df.columns), None
        )

        if start_col and end_col:
            try:
                start_s = self._to_seconds(df[start_col])
                end_s   = self._to_seconds(df[end_col])
                span    = end_s.max() - start_s.min()
                stats["time_span_h"] = round(
                    float(span) / 3600, 1
                )
            except Exception:
                pass

        for col in ["num_gpus", "gpus", "nGPUs"]:
            if col in df.columns:
                try:
                    stats["max_gpus"] = int(df[col].max())
                except Exception:
                    pass
                break

        return stats

    def generate(self):
        """
        Process RAPS job table into a power time series.
        Returns same format as AILoadProfile.generate()
        """
        df_jobs = self.load().copy()

        # ── Find column names ──────────────────────────────────
        start_col = next(
            (c for c in ["start_time", "start", "submit_time"]
             if c in df_jobs.columns), None
        )
        end_col = next(
            (c for c in ["end_time", "end", "finish_time"]
             if c in df_jobs.columns), None
        )
        gpu_col = next(
            (c for c in ["num_gpus", "gpus", "nGPUs"]
             if c in df_jobs.columns), None
        )

        # ── Convert times to float seconds ────────────────────
        if start_col:
            df_jobs["_start_s"] = self._to_seconds(
                df_jobs[start_col]
            )
            df_jobs["_start_s"] -= df_jobs["_start_s"].min()
        else:
            df_jobs["_start_s"] = 0.0

        if end_col:
            df_jobs["_end_s"] = self._to_seconds(
                df_jobs[end_col]
            )
            df_jobs["_end_s"] -= df_jobs["_end_s"].min()
        else:
            df_jobs["_end_s"] = float(self.duration_s)

        # ── Convert GPU count to float ─────────────────────────
        if gpu_col:
            df_jobs["_gpus"] = pd.to_numeric(
                df_jobs[gpu_col], errors="coerce"
            ).fillna(0)
        else:
            df_jobs["_gpus"] = 0.0

        # ── Build time array ───────────────────────────────────
        time_s = np.arange(0, self.duration_s, self.dt)
        power  = np.zeros(len(time_s))

        # ── Scale factor ───────────────────────────────────────
        max_gpus = float(df_jobs["_gpus"].max()) \
            if gpu_col else 1.0
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
