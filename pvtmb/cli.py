"""Command line interface.

    pvtmb pvt --api 35 --gas-gravity 0.75 --temperature 200 --rsb 600
    pvtmb zfactor --pressure 3000 --temperature 180 --gas-gravity 0.7
    pvtmb mbal history.csv --swi 0.24 --cw 3.62e-6 --cf 4.95e-6
    pvtmb validate
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import units
from .correlations import gas
from .diagnostics import identify_drive
from .fluid import TABLE_UNITS, BlackOil
from .material_balance import MaterialBalance


def _add_fluid_args(p: argparse.ArgumentParser, required: bool) -> None:
    g = p.add_argument_group("fluid")
    g.add_argument("--api", type=float, required=required, help="stock-tank oil gravity, degAPI")
    g.add_argument("--gas-gravity", type=float, required=required, help="gas specific gravity (air = 1)")
    g.add_argument("--temperature", type=float, required=required, help="reservoir temperature")
    g.add_argument("--temperature-unit", default="degF", help="degF, degC, K or degR (default degF)")
    g.add_argument("--rsb", type=float, help="solution GOR at the bubble point")
    g.add_argument("--gor-unit", default="scf/stb", help="scf/stb or sm3/sm3 (default scf/stb)")
    g.add_argument("--pb", type=float, help="bubble-point pressure (instead of --rsb)")
    g.add_argument("--correlation", default="standing", choices=["standing", "vasquez_beggs"])
    g.add_argument("--z-method", default="dak", choices=["dak", "hall_yarborough"])


def _fluid_from_args(a, pressure_unit: str) -> BlackOil:
    if (a.rsb is None) == (a.pb is None):
        raise SystemExit("error: give exactly one of --rsb or --pb")
    return BlackOil.from_units(api=a.api, gas_gravity=a.gas_gravity, temperature=a.temperature,
                               temperature_unit=a.temperature_unit, rsb=a.rsb, gor_unit=a.gor_unit,
                               pb=a.pb, pressure_unit=pressure_unit, correlation=a.correlation,
                               z_method=a.z_method)


def cmd_pvt(a) -> int:
    fluid = _fluid_from_args(a, a.pressure_unit)
    pmax = a.pmax if a.pmax is not None else units.convert(max(5000.0, 1.3 * fluid.pb), "psia", a.pressure_unit)
    pmin = a.pmin if a.pmin is not None else units.convert(100.0, "psia", a.pressure_unit)
    table = fluid.table(np.linspace(pmin, pmax, a.steps), pressure_unit=a.pressure_unit, units_out=a.units)
    print(fluid.describe())
    unit_row = {c: (TABLE_UNITS[c][1 if a.units == "si" else 0] or "-") for c in table}
    print(pd.concat([pd.DataFrame([unit_row]), table.map(lambda v: f"{v:.5g}")]).to_string(index=False))
    if a.output:
        table.to_csv(a.output, index=False)
        print(f"\nwrote {a.output}")
    if a.plot:
        from .plots import plot_pvt

        field_table = fluid.table(np.linspace(pmin, pmax, a.steps), pressure_unit=a.pressure_unit)
        plot_pvt(field_table, pb=fluid.pb, title=fluid.describe(), path=a.plot)
        print(f"wrote {a.plot}")
    return 0


def cmd_zfactor(a) -> int:
    if a.tpr is not None and a.ppr is not None:
        tpr, ppr = a.tpr, a.ppr
    elif None not in (a.pressure, a.temperature, a.gas_gravity):
        p = units.convert(a.pressure, a.pressure_unit, "psia")
        t = units.convert(a.temperature, a.temperature_unit, "degF")
        tpr, ppr = gas.pseudo_reduced(p, t, a.gas_gravity, a.h2s, a.co2)
        tpr, ppr = float(tpr), float(ppr)
    else:
        raise SystemExit("error: give --tpr and --ppr, or --pressure, --temperature and --gas-gravity")
    print(f"Tpr = {tpr:.4f}   ppr = {ppr:.4f}")
    print(f"z (Hall-Yarborough)         = {gas.z_hall_yarborough(tpr, ppr):.5f}")
    print(f"z (Dranchuk-Abou-Kassem)    = {gas.z_dranchuk_abou_kassem(tpr, ppr):.5f}")
    return 0


def _read_history(a) -> pd.DataFrame:
    df = pd.read_csv(a.history).rename(columns=str.lower)
    df["p"] = units.convert(df["p"].to_numpy(), a.pressure_unit, "psia")
    for col in ("np", "wp", "winj"):
        if col in df:
            df[col] = units.convert(df[col].to_numpy(), a.liquid_unit, "stb")
    for col in ("gp", "ginj"):
        if col in df:
            df[col] = units.convert(df[col].to_numpy(), a.gas_unit, "scf")
    return df


def cmd_mbal(a) -> int:
    history = _read_history(a)
    fluid = _fluid_from_args(a, a.pressure_unit) if a.api is not None else None
    cw = units.convert(a.cw, a.compressibility_unit, "1/psi")
    cf = units.convert(a.cf, a.compressibility_unit, "1/psi")
    mb = MaterialBalance(history, swi=a.swi, cw=cw, cf=cf, fluid=fluid)
    m = a.m if a.m == "estimate" else float(a.m)
    aquifer = {"none": None}.get(a.aquifer, a.aquifer)
    diag = identify_drive(mb, m=m, aquifer=aquifer)
    print(diag.report())

    if a.plots:
        from . import plots

        out = Path(a.plots)
        out.mkdir(parents=True, exist_ok=True)
        plots.plot_havlena_odeh(diag.result, path=out / "havlena_odeh.png")
        plots.plot_campbell(diag, path=out / "campbell.png")
        plots.plot_drive_indices(diag, path=out / "drive_indices.png")
        print(f"\nwrote plots to {out}/")
    if a.terms:
        mb.terms.to_csv(a.terms, index=False)
        print(f"wrote {a.terms}")
    return 0


def cmd_validate(a) -> int:
    from .validation import report

    print(report())
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="pvtmb", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("pvt", help="black-oil PVT table from correlations")
    _add_fluid_args(p, required=True)
    p.add_argument("--pressure-unit", default="psia")
    p.add_argument("--pmin", type=float, help="lowest table pressure (default 100 psia)")
    p.add_argument("--pmax", type=float, help="highest table pressure")
    p.add_argument("--steps", type=int, default=15)
    p.add_argument("--units", choices=["field", "si"], default="field", help="output units")
    p.add_argument("-o", "--output", help="write the table to CSV")
    p.add_argument("--plot", help="write a PVT figure (PNG)")
    p.set_defaults(func=cmd_pvt)

    p = sub.add_parser("zfactor", help="gas z-factor by both methods")
    p.add_argument("--tpr", type=float)
    p.add_argument("--ppr", type=float)
    p.add_argument("--pressure", type=float)
    p.add_argument("--pressure-unit", default="psia")
    p.add_argument("--temperature", type=float)
    p.add_argument("--temperature-unit", default="degF")
    p.add_argument("--gas-gravity", type=float)
    p.add_argument("--h2s", type=float, default=0.0, help="H2S mole fraction")
    p.add_argument("--co2", type=float, default=0.0, help="CO2 mole fraction")
    p.set_defaults(func=cmd_zfactor)

    p = sub.add_parser("mbal", help="Havlena-Odeh OOIP and drive mechanism from a history CSV")
    p.add_argument("history", help="CSV with p, np and optionally gp, wp, bo, rs, bg, bw, t (days)")
    p.add_argument("--swi", type=float, required=True, help="connate water saturation (fraction)")
    p.add_argument("--cw", type=float, default=3e-6, help="water compressibility")
    p.add_argument("--cf", type=float, default=4e-6, help="formation compressibility")
    p.add_argument("--compressibility-unit", default="1/psi")
    p.add_argument("--m", default="0", help="gas-cap ratio, or 'estimate'")
    p.add_argument("--aquifer", default="auto", choices=["auto", "none", "pot", "schilthuis"])
    p.add_argument("--pressure-unit", default="psia", help="unit of the p column (and --pb)")
    p.add_argument("--liquid-unit", default="stb", help="unit of np / wp columns, e.g. mstb, m3")
    p.add_argument("--gas-unit", default="scf", help="unit of the gp column, e.g. mscf")
    p.add_argument("--plots", help="directory for Havlena-Odeh / Campbell / drive-index plots")
    p.add_argument("--terms", help="write the F, Eo, Eg, Ef table to CSV")
    _add_fluid_args(p, required=False)
    p.set_defaults(func=cmd_mbal)

    p = sub.add_parser("validate", help="compare against textbook examples")
    p.set_defaults(func=cmd_validate)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, units.UnitError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
