import numpy as np

from pvtmb import plots
from pvtmb.datasets import VIRGINIA_HILLS, VIRGINIA_HILLS_PARAMS
from pvtmb.diagnostics import identify_drive
from pvtmb.fluid import BlackOil
from pvtmb.material_balance import MaterialBalance
from pvtmb.synthetic import synthetic_history


def test_pvt_plot(tmp_path):
    fluid = BlackOil(api=35.0, gas_gravity=0.75, temperature=200.0, rsb=600.0)
    out = plots.plot_pvt(fluid.table(np.linspace(200, 5000, 25)), pb=fluid.pb, path=tmp_path / "pvt.png")
    assert out.exists() and out.stat().st_size > 10_000


def test_material_balance_plots(tmp_path):
    mb = MaterialBalance(VIRGINIA_HILLS, **VIRGINIA_HILLS_PARAMS)
    d = identify_drive(mb)
    for fn, name in [(plots.plot_havlena_odeh, "ho.png"), (plots.plot_campbell, "camp.png"),
                     (plots.plot_drive_indices, "idx.png")]:
        arg = d.result if fn is plots.plot_havlena_odeh else d
        out = fn(arg, path=tmp_path / name)
        assert out.exists()


def test_all_havlena_odeh_forms_plot(tmp_path):
    fluid = BlackOil(api=32.0, gas_gravity=0.78, temperature=190.0, pb=3000.0)
    p = np.linspace(3000, 1800, 10)
    t = np.linspace(0, 2000, 10)
    hist = synthetic_history(fluid, N=50e6, m=0.3, pressures=p, t=t, swi=0.2)
    mb = MaterialBalance(hist, swi=0.2)
    for i, kwargs in enumerate([{"m": 0.3}, {"m": "estimate"},
                                {"m": 0.3, "aquifer": "schilthuis"},
                                {"m": "estimate", "aquifer": "pot"}]):
        plots.plot_havlena_odeh(mb.fit(**kwargs), path=tmp_path / f"{i}.png")
