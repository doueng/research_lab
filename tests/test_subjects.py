"""Subjects through the real CLI entry point: each subject gets its own seeds and dataroom."""

import shutil
import sys
from pathlib import Path

import pytest

from lab import cli, db

REPO_SEEDS = Path(__file__).parent.parent / "seeds"


def lab(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["lab", *argv])
    cli.main()


def write_seeds(name, objective, target):
    Path("seeds").mkdir(exist_ok=True)
    Path(f"seeds/{name}.toml").write_text(
        f'[lab]\nobjective = "{objective}"\n\n[[lead]]\ntarget = "{target}"\nreason = "r"\npriority = 60\n')


def test_subjects_have_separate_datarooms(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LAB_SUBJECT", raising=False)
    write_seeds("memory", "learn memory", "https://memory-seed")
    write_seeds("london", "learn london", "https://london-seed")

    lab(monkeypatch, "-s", "memory", "seed")
    lab(monkeypatch, "--subject", "london", "seed")
    lab(monkeypatch, "-s", "london", "lead", "https://thames-tunnel", "why", "90")

    mem, lon = db.connect("data/memory.db"), db.connect("data/london.db")
    assert db.get_meta(mem, "objective") == "learn memory"
    assert db.get_meta(lon, "objective") == "learn london"
    assert [x["target"] for x in db.frontier(mem)] == ["https://memory-seed"]
    assert [x["target"] for x in db.frontier(lon)] == ["https://thames-tunnel", "https://london-seed"]

    capsys.readouterr()
    lab(monkeypatch, "subjects")
    assert capsys.readouterr().out.split() == ["london", "seeded", "memory", "seeded"]


def test_default_subject_comes_from_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_seeds("hypertrophy", "grow", "https://h")
    monkeypatch.setenv("LAB_SUBJECT", "hypertrophy")
    lab(monkeypatch, "seed")
    assert db.get_meta(db.connect("data/hypertrophy.db"), "objective") == "grow"


def test_unknown_subject_is_refused_without_creating_a_dataroom(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_seeds("memory", "m", "https://m")
    with pytest.raises(SystemExit, match="unknown subject 'memroy'.*Known: memory"):
        lab(monkeypatch, "-s", "memroy", "status")
    assert not Path("data/memroy.db").exists()


@pytest.mark.parametrize("seeds", sorted(REPO_SEEDS.glob("*.toml")), ids=lambda p: p.stem)
def test_shipped_seed_files_load(seeds, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    Path("seeds").mkdir()
    shutil.copy(seeds, "seeds")
    lab(monkeypatch, "-s", seeds.stem, "seed")
    conn = db.connect(f"data/{seeds.stem}.db")
    assert db.get_meta(conn, "objective")
    assert len(db.frontier(conn)) >= 3
    assert "0 already present" in capsys.readouterr().out
