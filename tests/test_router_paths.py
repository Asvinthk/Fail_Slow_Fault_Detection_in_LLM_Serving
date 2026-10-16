"""Host-only regression test: Router.load_requests accepts absolute paths.

router.py unconditionally prefixed '../' (sim chdir's into astra-sim/ at
runtime), which turned absolute paths like 'C:\\...' into invalid
'../C:\\...' (OSError: Errno 22). Relative paths keep the old behaviour.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from serving.core.router import Router


def _router(req_num=0):
    return Router(num_instances=1, schedulers=[], req_num=req_num)


def _write(path, n=3):
    with open(path, "w") as f:
        for i in range(n):
            f.write(json.dumps({
                "input_toks": 32, "output_toks": 16,
                "arrival_time_ns": i * 1_000_000_000,
            }) + "\n")


def test_absolute_path_loads(tmp_path):
    wl = tmp_path / "w.jsonl"
    _write(str(wl))
    r = _router()
    r.load_requests(os.path.abspath(str(wl)))
    assert len(r._pending_requests) == 3


def test_relative_path_keeps_dotdot_prefix(tmp_path, monkeypatch):
    # Simulate runtime cwd (one dir below repo root, like astra-sim/):
    # load_requests('x.jsonl') must resolve to '../x.jsonl', i.e. tmp_path.
    _write(str(tmp_path / "x.jsonl"))
    sim_cwd = tmp_path / "sim"
    sim_cwd.mkdir()
    monkeypatch.chdir(str(sim_cwd))
    r = _router()
    r.load_requests("x.jsonl")
    assert len(r._pending_requests) == 3
