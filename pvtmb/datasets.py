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
