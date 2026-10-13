"""Textbook datasets used to validate the correlations and material balance.

Values are transcribed from the cited sources; units are field units.
"""

from __future__ import annotations

import pandas as pd

# Six crude-oil systems with measured bubble-point properties, used in
# Ahmed, T. "Reservoir Engineering Handbook" (ch. 2) to compare Standing,
# Vasquez-Beggs, Glaso, Marhoun and Petrosky-Farshad.
# Columns: temp (degF), pb (psia), Rsb (scf/STB), Bob (rb/STB), API, gas gravity
AHMED_CRUDES = pd.DataFrame(
    [
        (250.0, 2377.0, 751.0, 1.528, 47.1, 0.851),
        (220.0, 2620.0, 768.0, 1.474, 40.7, 0.855),
        (260.0, 2051.0, 693.0, 1.529, 48.6, 0.911),
        (237.0, 2884.0, 968.0, 1.619, 40.5, 0.898),
        (218.0, 3065.0, 943.0, 1.570, 44.2, 0.781),
        (180.0, 4239.0, 807.0, 1.385, 27.3, 0.848),
    ],
    columns=["temp", "pb", "rsb", "bob", "api", "gas_gravity"],
    index=pd.Index(range(1, 7), name="oil"),
)


# Virginia Hills Beaverhill Lake field: a volumetric, undersaturated oil
# reservoir (Ahmed, "Reservoir Engineering Handbook", ch. 11, Havlena-Odeh
# example). pb = 1500 psia, Swi = 0.24, cw = 3.62e-6 1/psi,
# cf = 4.95e-6 1/psi, Bw = 1.0. Published answer: N ~ 257 MMSTB (the
# volumetric estimate is 270.6 MMSTB).
# Production is converted from MSTB to STB.
VIRGINIA_HILLS = pd.DataFrame({
    "p": [3685, 3680, 3676, 3667, 3664, 3640, 3605, 3567, 3515, 3448, 3360, 3275, 3188],
    "np": [x * 1e3 for x in (0.0, 20.481, 34.750, 78.557, 101.846, 215.681, 364.613,
                             542.985, 841.591, 1273.530, 1691.887, 2127.077, 2575.330)],
    "wp": [x * 1e3 for x in (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                             0.159, 0.805, 2.579, 5.008, 6.500, 8.000)],
    "bo": [1.3102, 1.3104, 1.3104, 1.3105, 1.3105, 1.3109, 1.3116,
           1.3122, 1.3128, 1.3130, 1.3150, 1.3160, 1.3170],
}).astype(float)
VIRGINIA_HILLS_PARAMS = {"swi": 0.24, "cw": 3.62e-6, "cf": 4.95e-6}
