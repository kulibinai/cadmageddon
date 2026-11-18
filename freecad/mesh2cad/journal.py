"""A record of what every instrument call cost.

A4 says the loop must be cheap, and until this existed nothing measured it. The
reconstruction quality has been reported all along; the price of it -- calls,
output the model has to read, wall-clock -- was asserted and never counted.

One line per invocation, appended. Written by the CLI itself rather than by a
wrapper, so a run cannot be measured only when someone remembers to measure it.
"""

import json
import os
import time
from pathlib import Path

from mesh2cad import summary

# Rough, and labelled as rough wherever it is shown. What matters for A4 is the
# trend across runs and the relative cost of one instrument against another,
# both of which survive a constant-factor error in the conversion.
CHARS_PER_TOKEN = 4.0

DEFAULT_PATH = Path("out/journal.jsonl")


def path():
    return Path(os.environ.get("MESH2CAD_JOURNAL", DEFAULT_PATH))


class Tee:
    """Pass stdout through untouched while counting what went past."""

    def __init__(self, stream):
        self._s = stream
        self.chars = 0
        self._seen = []

    def write(self, text):
        self.chars += len(text)
        self._seen.append(text)
        return self._s.write(text)

    def text(self):
        return "".join(self._seen)

    def flush(self):
        return self._s.flush()

    def __getattr__(self, name):
        return getattr(self._s, name)


def record(entry):
    p = path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a") as fh:
            fh.write(json.dumps(entry) + "\n")
    except OSError:
        # Never let bookkeeping take down a reconstruction.
        pass


def run_and_record(fn, args, argv):
    """Time a command, count its output, and append a journal line."""
    import sys

    tee = Tee(sys.stdout)
    started = time.time()
    ok, err = True, None
    real, sys.stdout = sys.stdout, tee
    try:
        fn(args)
        # Only for a command that finished: a failed one has its error instead.
        tee.write(summary.block(getattr(args, "cmd", None), tee.text()))
    except BaseException as exc:                                   # noqa: BLE001
        ok, err = False, f"{type(exc).__name__}: {exc}"
        raise
    finally:
        sys.stdout = real
        record({
            "t": round(started, 3),
            "cmd": getattr(args, "cmd", None),
            "mesh": getattr(args, "mesh", None),
            "out": getattr(args, "out", None),
            "argv": argv,
            "seconds": round(time.time() - started, 3),
            "chars": tee.chars,
            "tokens_est": round(tee.chars / CHARS_PER_TOKEN),
            "ok": ok,
            "error": err,
        })
