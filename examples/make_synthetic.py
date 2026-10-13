"""Generate a synthetic water-drive history with a known answer.

A 30 API oil (pb = 3200 psia) is depleted from 4000 to 2000 psia over ten
years with a Schilthuis steady-state aquifer. The material balance in
`pvtmb mbal` should recover N = 90 MMSTB and C = 60 rb/(psi day).

    python examples/make_synthetic.py
"""

from pathlib import Path

import numpy as np

from pvtmb.aquifer import influx_function
from pvtmb.fluid import BlackOil
from pvtmb.synthetic import synthetic_history

N_TRUE = 90e6
C_TRUE = 60.0

fluid = BlackOil(api=30.0, gas_gravity=0.8, temperature=180.0, pb=3200.0)
t = np.linspace(0.0, 3650.0, 21)
# pressure decline that flattens as the aquifer responds
p = 2000.0 + 2000.0 * np.exp(-t / 1800.0) - 2000.0 * np.exp(-3650.0 / 1800.0) * t / 3650.0
hist = synthetic_history(fluid, N=N_TRUE, pressures=p, t=t, swi=0.22, cw=3e-6, cf=5e-6,
                         we=C_TRUE * influx_function("schilthuis", p, t))

# write only what a field would report: no PVT columns, volumes in MSTB / MMscf
out = hist[["t", "p", "np", "gp", "wp"]].copy()
out["np"] /= 1e3
out["gp"] /= 1e6
out = out.round({"t": 1, "p": 1, "np": 3, "gp": 3})
path = Path(__file__).with_name("water_drive_synthetic.csv")
out.to_csv(path, index=False)
print(f"wrote {path} ({len(out)} rows); true N = {N_TRUE / 1e6:.0f} MMSTB, C = {C_TRUE}")
