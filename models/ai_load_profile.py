import numpy as np
import pandas as pd

class AILoadProfile:
    def __init__(self, it_load_mw, num_gpus, gpu_power_w,
                 ramp_rate_pct, duration_min):
        self.it_load_mw    = it_load_mw
        self.num_gpus      = num_gpus
        self.gpu_power_w   = gpu_power_w
        self.ramp_rate_pct = ramp_rate_pct   # % IT load per second (NVIDIA: 20%)
        self.duration_s    = duration_min * 60
        self.dt            = 1               # 1-second timestep

    def generate(self):
        t        = np.arange(0, self.duration_s, self.dt)
        load     = np.zeros(len(t))
        events   = []

        # Start at idle (10% load)
        load[0]  = self.it_load_mw * 0.10
        ramp_mw_per_s = self.it_load_mw * (self.ramp_rate_pct / 100.0)

        # Simulate multiple AI job start/stop events
        np.random.seed(42)
        job_starts = sorted(np.random.choice(
            range(30, self.duration_s - 120, 60),
            size=max(2, self.duration_s // 300),
            replace=False
        ))

        current_load   = self.it_load_mw * 0.10
        job_active     = False
        job_end_time   = 0
        drop_events    = []

        for i in range(1, len(t)):
            time_now = t[i]

            # Check if a job starts
            for js in job_starts:
                if abs(time_now - js) < 1 and not job_active:
                    job_duration  = np.random.randint(120, 400)
                    job_end_time  = time_now + job_duration
                    job_active    = True
                    break

            if job_active:
                # Ramp up to full load
                target = self.it_load_mw * np.random.uniform(0.7, 1.0)
                if current_load < target:
                    current_load = min(current_load + ramp_mw_per_s, target)

                # Job ends → sudden drop
                if time_now >= job_end_time:
                    drop_start   = current_load
                    current_load = self.it_load_mw * 0.10
                    job_active   = False
                    drop_events.append({
                        "start":      float(time_now),
                        "end":        float(time_now + 5),
                        "drop_mw":    round(drop_start - current_load, 2),
                        "drop_pct":   round((drop_start - current_load)
                                            / self.it_load_mw * 100, 1),
                        "ramp_mw_s":  round((drop_start - current_load) / 5, 2)
                    })
            else:
                # Idle with small noise
                current_load = max(
                    self.it_load_mw * 0.10,
                    current_load + np.random.uniform(-0.5, 0.5)
                )

            load[i] = max(0, min(current_load, self.it_load_mw))

        df = pd.DataFrame({"time_s": t, "it_load_mw": load})
        return df, drop_events
