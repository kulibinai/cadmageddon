"""A time budget for the commands that can run into the 1800 s command limit.

On dense or many-body meshes `section`, `section --face`, `ortho --frame`,
`residual` and `sketch --op chamfer/fillet` can each run past the agent's
30-minute limit and print nothing, costing the model the call and the time.
A command under a budget runs in a child process the CLI can stop; the slow
steps say what they are doing (`stage`), and a stop says which one ran out of
time and what to do.

A signal in the same process would not do: the slow step is a single call
into GEOS, OCC or manifold, and a Python handler runs only once it returns.
"""

import os
import signal
import sys
import tempfile
import time
from pathlib import Path

STAGE_ENV = "M2C_STAGE_FILE"
BUDGET_ENV = "M2C_TIME_BUDGET"
# Seconds a budgeted command may run: a sixth of the 1800 s command limit,
# so a refusal leaves the run its time. $M2C_TIME_BUDGET overrides it; 0
# means no budget.
TIME_BUDGET = 300.0


def stage(text):
    """Say what the command is doing now, for a stop to name."""
    path = os.environ.get(STAGE_ENV)
    if path:
        try:
            Path(path).write_text(text)
        except OSError:
            pass


def budget():
    raw = os.environ.get(BUDGET_ENV)
    if raw is None:
        return TIME_BUDGET
    try:
        val = float(raw)
    except ValueError:
        return TIME_BUDGET
    return None if val <= 0 else val


def run(work, seconds, what, advice):
    """`work()` in a child, stopped after `seconds`: its exit code, or a
    refusal naming the stage it was in."""
    fd, path = tempfile.mkstemp(prefix="m2c_stage_")
    os.close(fd)
    os.environ[STAGE_ENV] = path
    sys.stdout.flush()
    sys.stderr.flush()
    pid = os.fork()
    if pid == 0:                                     # the child does the work
        code = 0
        try:
            work()
        except SystemExit as exc:
            code = exc.code
            if isinstance(code, str):
                sys.stderr.write(code + "\n")
                code = 1
            code = code or 0
        except BaseException:                       # noqa: BLE001
            import traceback
            traceback.print_exc()
            code = 1
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(int(code))
    t0 = time.time()
    try:
        while True:
            done, status = os.waitpid(pid, os.WNOHANG)
            if done:
                return os.waitstatus_to_exitcode(status)
            if time.time() - t0 > seconds:
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
                now = Path(path).read_text().strip() if Path(path).is_file() else ""
                sys.exit(f"{what} stopped after {seconds:.0f} s"
                         + (f", while {now}" if now else "")
                         + f" -- on the 1800 s command limit it would have run out with "
                           f"nothing to show. {advice} (${BUDGET_ENV}=<seconds> allows "
                           f"longer; 0 means no limit)")
            time.sleep(0.2)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
        os.environ.pop(STAGE_ENV, None)
