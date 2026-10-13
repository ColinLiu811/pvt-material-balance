from pathlib import Path

import pandas as pd
import pytest

from pvtmb.cli import main
from pvtmb.datasets import VIRGINIA_HILLS

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def test_pvt_table_field_and_si(capsys, tmp_path):
    out = tmp_path / "pvt.csv"
    assert main(["pvt", "--api", "35", "--gas-gravity", "0.75", "--temperature", "200",
                 "--rsb", "600", "--steps", "5", "-o", str(out)]) == 0
    text = capsys.readouterr().out
    assert "pb=2570 psia" in text and "scf/stb" in text
    assert len(pd.read_csv(out)) == 6  # 5 steps + bubble point

    assert main(["pvt", "--api", "35", "--gas-gravity", "0.75", "--temperature", "93.3",
                 "--temperature-unit", "degC", "--pb", "17.72", "--pressure-unit", "MPa",
                 "--pmin", "1", "--pmax", "30", "--units", "si"]) == 0
    text = capsys.readouterr().out
    assert "sm3/sm3" in text and "MPa" in text


def test_pvt_requires_one_of_rsb_pb():
    with pytest.raises(SystemExit):
        main(["pvt", "--api", "35", "--gas-gravity", "0.75", "--temperature", "200"])


def test_zfactor(capsys):
    assert main(["zfactor", "--tpr", "1.5", "--ppr", "2"]) == 0
    out = capsys.readouterr().out
    assert "0.82" in out
    assert main(["zfactor", "--pressure", "20", "--pressure-unit", "MPa", "--temperature", "80",
                 "--temperature-unit", "degC", "--gas-gravity", "0.7", "--co2", "0.05"]) == 0


def test_mbal_virginia_hills_with_unit_conversion(capsys, tmp_path):
    csv = tmp_path / "vh.csv"
    df = VIRGINIA_HILLS.copy()
    df["np"] /= 1e3
    df["wp"] /= 1e3
    df.to_csv(csv, index=False)
    assert main(["mbal", str(csv), "--swi", "0.24", "--cw", "3.62e-6", "--cf", "4.95e-6",
                 "--liquid-unit", "mstb", "--plots", str(tmp_path / "plots"),
                 "--terms", str(tmp_path / "terms.csv")]) == 0
    out = capsys.readouterr().out
    assert "282.8" in out
    assert "undersaturated oil expansion" in out
    assert (tmp_path / "plots" / "havlena_odeh.png").exists()
    assert "F" in pd.read_csv(tmp_path / "terms.csv")


def test_mbal_synthetic_water_drive_example(capsys):
    assert main(["mbal", str(EXAMPLES / "water_drive_synthetic.csv"), "--swi", "0.22",
                 "--cw", "3e-6", "--cf", "5e-6", "--liquid-unit", "mstb", "--gas-unit", "mmscf",
                 "--api", "30", "--gas-gravity", "0.8", "--temperature", "180", "--pb", "3200"]) == 0
    out = capsys.readouterr().out
    assert "water drive" in out
    assert "OOIP N      : 90.0" in out  # rounding in the CSV keeps us within 0.05 MMSTB


def test_mbal_reports_input_errors(capsys, tmp_path):
    csv = tmp_path / "bad.csv"
    pd.DataFrame({"p": [3000, 2900, 2800], "np": [5, 10, 20], "bo": 1.3}).to_csv(csv, index=False)
    assert main(["mbal", str(csv), "--swi", "0.2"]) == 2
    assert "initial" in capsys.readouterr().err


def test_validate(capsys):
    assert main(["validate"]) == 0
    out = capsys.readouterr().out
    assert "Virginia Hills" in out and "AARE" in out
