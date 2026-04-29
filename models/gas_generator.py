import numpy as np

class GasGenerator:
    def __init__(self, rated_mw, droop_pct,
                 governor_time_s, min_load_pct):
        self.rated_mw        = rated_mw
        self.droop           = droop_pct / 100.0
        self.gov_time        = governor_time_s   # seconds to respond
        self.min_load_mw     = rated_mw * (min_load_pct / 100.0)
        self.nominal_freq    = 50.0              # Hz
        self.dt              = 1                 # 1-second timestep

    def simulate(self, load_profile_mw, bess_power_mw):
        """
        Simulate generator response given IT load and BESS support.
        bess_power_mw: positive = BESS charging (absorbing surplus)
                       negative = BESS discharging (supplying load)
        """
        n               = len(load_profile_mw)
        gen_output      = np.zeros(n)
        freq_deviation  = np.zeros(n)
        overspeed_flag  = np.zeros(n, dtype=bool)

        # Generator starts at rated output
        gen_output[0]   = self.rated_mw * 0.8

        for i in range(1, n):
            # Net load seen by generator
            # BESS positive = charging = acts as extra load
            # BESS negative = discharging = reduces load on generator
            net_load = load_profile_mw[i] + bess_power_mw[i]
            net_load = max(self.min_load_mw, net_load)

            # Governor response — first order lag
            alpha       = self.dt / (self.gov_time + self.dt)
            gen_output[i] = (alpha * net_load
                             + (1 - alpha) * gen_output[i - 1])
            gen_output[i] = np.clip(gen_output[i], self.min_load_mw,
                                    self.rated_mw * 1.05)

            # Frequency deviation due to load-generation mismatch
            # Δf = (P_gen - P_load) / (rated_mw * droop) * nominal_freq
            mismatch         = gen_output[i] - net_load
            freq_deviation[i] = ((mismatch / (self.rated_mw * self.droop))
                                 * self.nominal_freq * 0.01)
            freq_deviation[i] = np.clip(freq_deviation[i], -2.0, 2.0)

            # Overspeed detection (load rejection event)
            if (load_profile_mw[i] < load_profile_mw[i - 1] * 0.7
                    and abs(freq_deviation[i]) > 0.5):
                overspeed_flag[i] = True

        return gen_output, freq_deviation, overspeed_flag
