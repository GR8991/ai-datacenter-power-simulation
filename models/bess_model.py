import numpy as np

class BESSModel:
    def __init__(self, capacity_mwh, max_power_mw,
                 soc_initial_pct, soc_min_pct, soc_max_pct,
                 mode, dummy_load_mw):
        self.capacity_mwh   = capacity_mwh
        self.max_power_mw   = max_power_mw
        self.soc            = soc_initial_pct / 100.0
        self.soc_min        = soc_min_pct / 100.0
        self.soc_max        = soc_max_pct / 100.0
        self.mode           = mode            # GFM or GFL
        self.dummy_load_mw  = dummy_load_mw
        self.efficiency     = 0.95
        self.dt             = 1              # seconds

    def simulate(self, load_profile_mw, gen_rated_mw):
        """
        Simulate BESS response to AI load changes.
        Returns power flow, SOC, dummy load activation.
        """
        n                   = len(load_profile_mw)
        bess_power          = np.zeros(n)   # +charge / -discharge
        soc_arr             = np.zeros(n)
        dummy_active        = np.zeros(n)
        bess_charging       = np.zeros(n)

        soc_arr[0]          = self.soc
        prev_load           = load_profile_mw[0]

        for i in range(1, n):
            curr_load  = load_profile_mw[i]
            soc_now    = soc_arr[i - 1]
            load_delta = prev_load - curr_load   # positive = load dropped

            if self.mode == "Grid-Forming (GFM)":
                # ── GFM: BESS proactively absorbs surplus ──────────
                if load_delta > 0:
                    # Load dropped → surplus → charge BESS
                    surplus      = min(load_delta,
                                       self.max_power_mw,
                                       gen_rated_mw * 0.9 - curr_load)
                    surplus      = max(0, surplus)

                    if soc_now < self.soc_max:
                        # BESS can absorb
                        charge_mw        = min(surplus,
                                               (self.soc_max - soc_now)
                                               * self.capacity_mwh * 3600
                                               / self.dt)
                        charge_mw        = min(charge_mw, self.max_power_mw)
                        bess_power[i]    = charge_mw    # positive = charging
                        bess_charging[i] = charge_mw

                        # Remaining surplus goes to dummy load
                        remaining        = surplus - charge_mw
                        dummy_active[i]  = min(remaining, self.dummy_load_mw)
                    else:
                        # BESS full → all to dummy load
                        dummy_active[i]  = min(surplus, self.dummy_load_mw)
                        bess_power[i]    = 0

                elif load_delta < 0:
                    # Load spiked → discharge BESS
                    deficit          = min(abs(load_delta), self.max_power_mw)
                    if soc_now > self.soc_min:
                        bess_power[i] = -deficit   # negative = discharging
                    else:
                        bess_power[i] = 0

            else:
                # ── GFL: BESS waits for generator signal ───────────
                # Delayed response — 3 second lag
                if i > 3 and load_delta > 0:
                    surplus = min(load_delta * 0.5, self.max_power_mw)
                    if soc_now < self.soc_max:
                        bess_power[i]    = surplus * 0.5
                        bess_charging[i] = surplus * 0.5
                    # More goes to dummy load since GFL is slow
                    dummy_active[i] = min(load_delta - bess_power[i],
                                          self.dummy_load_mw)
                elif i > 3 and load_delta < 0:
                    if soc_now > self.soc_min:
                        bess_power[i] = -min(abs(load_delta) * 0.5,
                                             self.max_power_mw)

            # Update SOC
            power_mw       = bess_power[i]
            if power_mw > 0:  # charging
                delta_soc  = (power_mw * self.efficiency
                               * self.dt / 3600) / self.capacity_mwh
                soc_arr[i] = min(self.soc_max, soc_now + delta_soc)
            else:             # discharging
                delta_soc  = (abs(power_mw) / self.efficiency
                               * self.dt / 3600) / self.capacity_mwh
                soc_arr[i] = max(self.soc_min, soc_now - delta_soc)

            prev_load = curr_load

        return bess_power, soc_arr, dummy_active, bess_charging
