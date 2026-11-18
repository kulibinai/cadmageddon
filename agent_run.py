"""Reconstruct CAD programs from meshes with a VLM over an OpenAI-compatible endpoint.

    # one mesh
    python agent_run.py --mesh data/part.stl --out runs/ \\
        --base-url http://127.0.0.1:8000/v1 --model Qwen3.8-27B

    # every mesh in a directory, four parts at a time
    python agent_run.py --mesh data/meshes/ --out runs/ --jobs 4 \\
        --base-url http://127.0.0.1:8000/v1 --model Qwen3.8-27B

Each mesh gets its own directory, which is also the directory the agent's
commands run in:

    runs/<stem>/part.stl          the input mesh (the agent refers to it by this name)
    runs/<stem>/run/              the agent's work: reconstruction.py, trace.json,
                                  renders, outputs/ (every instrument output)
    runs/<stem>/attempt_<k>/      earlier attempts, when --attempts > 1
    runs/<stem>/<stem>.py         the delivered CadQuery program
    runs/<stem>/<stem>.stl        that program exported in the input mesh's frame
    runs/<stem>/agent.log         the console of the run
    runs/<stem>/metrics.json      status, calls, time, tokens, how the run ended

`evaluate.py --predictions runs/` scores that tree.

Everything the model is shown comes from the instruments: the text they print
and the PNGs they write. It never sees the mesh file, the ground truth, or the
score -- which is the same rule the contract enforces on instruments, now
applied to the agent as well.

THE MODEL'S OUTPUT IS UNTRUSTED INPUT. It is parsed for a single command and
that command must begin with `m2c.py`; it is executed as an argument vector with
no shell, so nothing it emits can be interpreted as shell syntax. A model that
asks for anything else gets told no and the loop continues -- that is a
measurement of the model, not an error in the harness.
"""

import argparse
import base64
import contextlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
M2C = ROOT / "scripts" / "m2c.py"
MESH_SUFFIXES = (".stl", ".obj", ".ply", ".off")
# The whole rest of the line: a work dir whose name has spaces puts them in
# every render path, and `\S+` would cut the path at the first one.
IMAGE_LINE = re.compile(r"^IMAGE:[ \t]*(\S.*?)[ \t]*$", re.M)
FENCE = re.compile(r"```(?:bash|sh|shell)?\s*\n(.*?)```", re.S)
# Only a safety limit for an unexpected command. `section --built` prints 3-4
# and a cap of 2 always dropped its third, the target-vs-built image.
MAX_IMAGES_PER_COMMAND = 8
# How many image-printing commands keep their renders, and how many renders that
# may come to. Older ones are replaced by a note. A render is about 1k tokens, so
# twelve are 5% of a 250k window; the window that filled five calls in was 32k.
# Both are bounds on the same walk back from the newest: K on how far a render
# reaches, N on its cost when several commands print 3-4 each.
KEEP_IMAGE_COMMANDS = 6
KEEP_IMAGES = 12
MAX_OUT_CHARS = 50000
# How many recent turns keep their reasoning whole. Older ones are compacted to
# the conclusion they reached and the command they ran.
#
# This is the same problem as the renders, and a larger one: on complex parts
# the model's own deliberation can outweigh the instrument output several times
# over, and a run that keeps all of it fills the context window long before the
# part is built.
#
# What is kept is the tail, because a reply that reasons its way to a choice puts
# the choice at the end, and the command fence verbatim so the record of what was
# actually run stays exact. The turn says it was compacted rather than leaving a
# silent hole.
KEEP_REASONING_FOR_TURNS = 2
REASONING_TAIL_CHARS = 800
# CONSECUTIVE turns that made no progress -- no command, or a command argparse
# rejected. Reset by every command that RUNS, because several non-commands in a
# row means stuck, while the same number spread over a long reconstruction is
# just a model that thinks between actions; counting them cumulatively would
# end long runs that still have budget left.
MAX_NUDGES = 6
# What one more message costs, fitted on consecutive pairs of calls: about 0.6
# tokens a character and 1000 an image. The whole prompt does not fit that line
# well (the error depends on what the text is), but the step from one prompt to
# the next does. So the next prompt is predicted from the last one the server
# measured, and the margin covers the worst miss of that step.
TOKENS_PER_CHAR = 0.6
TOKENS_PER_IMAGE = 1000
CONTEXT_MARGIN = 6000
CONTEXT_WINDOW = 250000
# How full the window may be once the ladder has run. Stopping just under the
# limit means paying for the ladder again on every following turn.
CONTEXT_TARGET = 0.70
# How many commands an output stays whole for. After that it is replaced, once,
# by its SUMMARY: block; tool text was the one thing nothing here ever shortened,
# 83% of the prompts that overflowed.
KEEP_OUTPUTS = 4
SUMMARY_BLOCK = re.compile(r"^SUMMARY:\n((?:  .*\n?)+)", re.M)
# A failed command has no summary. The end of what it said is the error.
ERROR_TAIL_CHARS = 400
# Below this a reply cannot hold a command and the reasoning before it.
MIN_REPLY_TOKENS = 1000
CONTEXT_ERROR = "maximum context length"
FIRST_MESSAGE = "Begin. You have not seen the part yet."


def encode(path):
    return base64.b64encode(Path(path).read_bytes()).decode()


def image_to_data_url(path):
    return f"data:image/png;base64,{encode(path)}"


# What to say immediately before a render, keyed by the file the instrument
# wrote. `overview` now emits two pictures of the same part rather than one
# composite, and two unlabelled images in a row is a guess the model should not
# have to make -- the four isometric views and the six orthographic ones answer
# different questions. Anything not named here is attached unlabelled, exactly
# as before.
IMAGE_LABELS = {
    "iso.png": "isometric views image:",
    "ortho.png": "orthographic views image:",
    # `progress` sends two sheets, one per solid, and which solid a sheet is of
    # decides what its colours mean: the paint is always the distance from THIS
    # mesh to the other one, so the same physical error is lime on the target
    # sheet and orange on the prediction sheet. Unlabelled, the two are
    # indistinguishable -- they are framed and scaled to be interchangeable on
    # purpose -- and a reader would have no way to tell missing material from
    # invented material.
    "target_iso.png": (
        "Isometric views for target. Lime color marks slight deviations from "
        "predict, dark green - large differences."),
    "predict_iso.png": (
        "Isometric views for predict. Orange color marks slight deviations "
        "from target, red - large differences."),
    "target_ortho.png": (
        "Orthographic views for target. Lime color marks slight deviations "
        "from predict, dark green - large differences."),
    "predict_ortho.png": (
        "Orthographic views for predict. Orange color marks slight deviations "
        "from target, red - large differences."),
}


def reasoning_tokens(usage):
    """How much of the completion was thinking, when the server says so.

    Qwen reasons before it answers, and the count is spelled two ways depending
    on the vLLM build: nested under `completion_tokens_details`, or flat on
    `usage`. Both are read, neither is assumed. A build that publishes neither
    gets None -- it did not report that the model thought nothing.
    """
    details = getattr(usage, "completion_tokens_details", None)
    if details is not None:
        n = getattr(details, "reasoning_tokens", None)
        if n is not None:
            return n
    return getattr(usage, "reasoning_tokens", None)


def read_usage(resp):
    """What one completion cost, as the server reported it, or None.

    `usage` is absent or None depending on how the endpoint is configured, and
    an unknown count is not a zero count -- reporting 0 prompt tokens claims a
    measurement nobody took. Missing stays missing, all the way into trace.json.
    """
    u = getattr(resp, "usage", None)
    if u is None:
        return None
    got = {"prompt": getattr(u, "prompt_tokens", None),
           "completion": getattr(u, "completion_tokens", None),
           "total": getattr(u, "total_tokens", None),
           # Reasoning is part of completion_tokens, not an addition to it.
           "reasoning": reasoning_tokens(u)}
    return got if any(v is not None for v in got.values()) else None


class TokenLedger:
    """Running totals over every response the run paid for.

    Every one, including the replies that get thrown away. A truncated turn is
    re-asked with a doubled budget, and the attempt that was cut off had still
    been generated and still been billed; a ledger fed at the log sites would
    miss exactly the turns that cost the most. So it is fed where the response
    arrives instead, before anything decides what to do with it.

    A key stays None until some response actually reports it, so "the server
    never said" and "it was zero" do not read the same.
    """

    KEYS = ("prompt", "completion", "total", "reasoning")

    def __init__(self):
        self.sums = dict.fromkeys(self.KEYS)
        self.responses = 0
        self.unmeasured = 0

    def add(self, resp):
        """Count one response, and hand back its usage for the turn's entry."""
        usage = read_usage(resp)
        self.responses += 1
        if usage is None:
            self.unmeasured += 1
            return None
        for k in self.KEYS:
            if usage[k] is not None:
                self.sums[k] = (self.sums[k] or 0) + usage[k]
        return usage

    def totals(self):
        """The top-level block: the sums, and how many responses are behind them."""
        return dict(self.sums, responses=self.responses,
                    responses_without_usage=self.unmeasured)

    def summary(self):
        """One console line, which says "unknown" where it means unknown."""
        if not self.responses:
            return "no responses came back, so there is nothing to count"
        if all(v is None for v in self.sums.values()):
            return (f"{self.responses} response(s), token counts unknown -- "
                    f"the endpoint reported no usage")
        say = {k: ("unknown" if v is None else v) for k, v in self.sums.items()}
        line = (f"{self.responses} response(s), {say['prompt']} prompt "
                f"+ {say['completion']} completion tokens")
        if self.sums["reasoning"] is not None:
            line += f", {say['reasoning']} of them reasoning"
        if self.unmeasured:
            line += (f"; {self.unmeasured} came back with no usage block, so "
                     f"the counts are a floor")
        return line


# "about" appears when the residual came from a voxel grid rather than a
# boolean -- the wording the harness itself prints for a mesh that will not
# close. Without it those lines parsed as having no volume at all.
RESIDUAL_VOL = re.compile(r"(?:MISSING|EXTRA)\s+(?:about\s+)?([0-9.eE+-]+)\s*mm\^3")
# How much better the last few checks have to be before the run counts as still
# making progress. Below this it is thrashing, not building.
PROGRESS_FRACTION = 0.05


def residual_total(line):
    """Total unexplained volume on a residual line, or None if it has none."""
    if not line:
        return None
    if "nothing above the sliver floor" in line:
        return 0.0
    got = RESIDUAL_VOL.findall(line)
    return sum(float(v) for v in got) if got else None


RESIDUAL_TARGET = re.compile(r"target volume(?: about)?\s+([0-9.eE+-]+)")
# What fraction of the part may be left over and still count as a concession.
# Below it, further building almost never improves the part; above it, there is
# real work left and DONE should still be refused.
CONCEDE_FRACTION = 0.01


def near_enough(line):
    """True when what is left is under CONCEDE_FRACTION of the part.

    The stall rule needs three checks to see a trend, and a simple part is often
    built in one or two features, so there is no trend to see. This lets such a
    part end once what is left is negligible.

    A preview describes a program that was never written, so it is never
    evidence that anything is near enough.
    """
    if not line or "[PREVIEW:" in line:
        return False
    left = residual_total(line)
    m = RESIDUAL_TARGET.search(line)
    if left is None or not m:
        return False
    target = float(m.group(1))
    return target > 0 and left <= target * CONCEDE_FRACTION


def stalled(history, window=3, fraction=PROGRESS_FRACTION):
    """True when the last `window` checks did not materially improve on the rest.

    Exact repetition is not enough: a model with two competing fixes oscillates
    between them rather than repeating one reading.

    Compares the best of the recent window against the best of everything
    before it, so a run that is still finding real improvements is never cut
    off, and one that is going round in circles is not asked to keep going.
    """
    vals = [v for v in history if v is not None]
    if len(vals) < window:
        return False
    recent = vals[-window:]
    before = vals[:-window]
    if not before:
        # Exactly `window` checks and nothing to compare them against. They are
        # evidence of a stall only if they are not improving among themselves --
        # three readings of the same number say the last two features changed
        # nothing.
        best, worst = min(recent), max(recent)
        return worst <= 0 or (worst - best) <= worst * fraction
    if min(before) <= 0:
        return True
    return min(recent) >= min(before) * (1.0 - fraction)


def improving(history, fraction=PROGRESS_FRACTION):
    """Did the last check materially beat the one before it?

    Deliberately only two readings, where `stalled` wants `window` of them.
    A run that has emitted twice and is stuck cannot produce a third reading
    without a third feature, so `stalled` stays False for want of history and
    says nothing about progress either way. This says the little it can: the
    number moved, or it did not.
    """
    vals = [v for v in history if v is not None]
    if len(vals) < 2:
        # One reading is not a trend. Treated as still improving, so the only
        # thing that can end a run this early is a residual small enough for
        # `near_enough` -- never a guess about a direction nobody has measured.
        return True
    prev, last = vals[-2], vals[-1]
    if prev <= 0:
        return False
    return last < prev * (1.0 - fraction)


def extract_command(text):
    """The one command the model wants run, or None.

    Prefers a fenced block; falls back to a line that looks like an invocation.
    Only `m2c.py ...` is ever returned -- see the module docstring.

    Returns (command, superseded) where `superseded` are the other distinct
    commands the reply contained, so the caller can say which one ran.

    The LAST one wins, not the first. The model reasons inside the reply, and
    while it reasons it drafts commands and then argues itself out of them --
    every draft in a fence, indistinguishable from the real thing. A reply that
    reasons its way to a choice puts the choice at the end.
    """
    blocks = [b.strip() for b in FENCE.findall(text or "")]
    candidates = []
    for b in blocks:
        candidates += [ln.strip() for ln in b.splitlines() if ln.strip()]
    candidates += [ln.strip() for ln in (text or "").splitlines()
                   if ln.strip().startswith("m2c.py")]

    found = []
    for c in candidates:
        c = c.lstrip("$ ").strip()
        if c.startswith("m2c.py") and c not in found:
            found.append(c)
    if not found:
        return None, []
    # Picking one of several is still a choice the model did not make, so the
    # caller is handed the rest to report rather than dropping them in silence.
    return found[-1], found[:-1]


def view_of(path):
    """Which view a render is of: its name without the level, so a section of XY
    replaces the last section of XY, and `_built_fill` stays a view of its own."""
    return re.sub(r"_[+-]?\d+(?:\.\d+)?", "", Path(path).stem)


def _take_image(entry, image):
    """Remove one render from its tool message. The label goes with the picture
    it introduces: left behind, an "isometric views image:" with nothing after it
    reads as a render that failed to arrive."""
    content = entry["message"]["content"]
    i = next(k for k, part in enumerate(content) if part is image["part"])
    del content[i]
    if i and content[i - 1].get("text") in IMAGE_LABELS.values():
        del content[i - 1]
    entry["images"].remove(image)


def attach_images(entry, paths, outputs):
    """Attach every render a command printed, each view once.

    A new render of a view replaces the older one, and a render whose bytes are
    already in the conversation is not sent again: the same file twice took two
    of the kept slots.
    """
    content = entry["message"]["content"]
    paths = [p for p in paths if Path(p).exists()]
    for p in paths[:MAX_IMAGES_PER_COMMAND]:
        name, view, url = Path(p).name, view_of(p), image_to_data_url(p)
        twin = next((o for o in outputs for i in o["images"]
                     if i["part"]["image_url"]["url"] == url), None)
        if twin:
            content.append({"type": "text", "text":
                            f"[{name} is identical to the render shown for "
                            f"command {twin['n']}]"})
            continue
        for o in outputs:
            if o is entry:
                continue
            for old in [i for i in o["images"] if i["view"] == view]:
                _take_image(o, old)
                o["message"]["content"].append({"type": "text", "text": (
                    f"[replaced: {old['name']}; the newer render of this view "
                    f"is at command {entry['n']}]")})
        label = IMAGE_LABELS.get(name)
        if label:
            content.append({"type": "text", "text": label})
        part = {"type": "image_url", "image_url": {"url": url}}
        content.append(part)
        entry["images"].append({"name": name, "view": view, "part": part})
    extra = len(paths) - MAX_IMAGES_PER_COMMAND
    if extra > 0:
        content.append({"type": "text", "text":
                        f"[{extra} further render(s) from this command are "
                        f"not attached; ask for them one at a time]"})


def drop_old_images(outputs, keep, keep_images):
    """Keep renders for the last `keep` commands that have any, `keep_images` in
    all, walking back from the newest and stopping at the first that passes
    either limit.

    A command's renders stay or go together, so a section is never split from its
    elevation, and the newest command's stay even if they alone pass the limit.
    What the note says is checked, not assumed: "your reading of them is above"
    was often false, the reasoning having been dropped and the reply empty. Every
    command that prints a render only reads, so naming it to re-run is safe.
    """
    shown = [o for o in outputs if o["images"]][::-1]
    total = 0
    for age, o in enumerate(shown):
        total += len(o["images"])
        if age and (age >= keep or total > keep_images):
            for old in shown[age:]:
                names = ", ".join(i["name"] for i in old["images"])
                for image in list(old["images"]):
                    _take_image(old, image)
                old["message"]["content"].append({"type": "text", "text": (
                    f"[expired: {names} of command {old['n']}; re-render with: "
                    f"{old['cmd']}]")})
            return


def elide(out, limit=MAX_OUT_CHARS):
    """Cut an over-long instrument output from the MIDDLE, and say so.

    It used to be `out[:limit]`, which keeps the head and drops the tail in
    silence. On a 26-tooth helical gear `section` printed 30,338 characters, so
    every one of the model's section calls lost the bore ring, the level table
    and the band table -- the tail is where an instrument puts its summary, and
    the summary is the plan. The model was left with a list of tooth vertices
    and no idea it was missing anything.

    R2 forbids an instrument truncating in silence. The harness was exempt only
    because nobody had looked at it.
    """
    if len(out) <= limit:
        return out
    head, tail = out[:limit * 2 // 3], out[-(limit // 3):]
    gone = len(out) - len(head) - len(tail)
    lines = out[len(head):len(out) - len(tail)].count("\n")
    return (f"{head}\n\n[{gone} characters ({lines} lines) dropped from the "
            f"MIDDLE of this output to fit the window. The head and the tail "
            f"are both shown. Re-run with a narrower request if you need what "
            f"was in between.]\n\n{tail}")


def compact_old_reasoning(messages, keep, tail=REASONING_TAIL_CHARS):
    """Cut all but the `keep` most recent replies down to decision and command.

    See KEEP_REASONING_FOR_TURNS. The model's deliberation is the largest thing
    in the conversation by a wide margin, and almost all of it has done its work
    by the time the next instrument has answered.

    Kept: the last `tail` characters of the prose, and every command fence
    verbatim. Dropped: the middle of the argument. The note says how much went,
    so a reader of the trace is never told less than the truth about what the
    model could still see.
    """
    idx = [i for i, m in enumerate(messages)
           if m.get("role") == "assistant" and isinstance(m.get("content"), str)]
    for i in idx[:-keep] if keep else idx:
        text = messages[i]["content"]
        fences = [f"```bash\n{b.strip()}\n```" for b in FENCE.findall(text)]
        prose = FENCE.sub("", text).strip()
        # Already compact, or already compacted: nothing to gain, and re-cutting
        # a compacted reply would eat the conclusion a character at a time.
        if len(text) <= tail or prose.startswith("[earlier reasoning"):
            continue
        cut = len(prose) - tail
        if cut > 0:
            prose = ("[earlier reasoning in this turn compacted -- "
                     f"{cut} characters dropped; the conclusion follows]\n"
                     + prose[-tail:])
        messages[i]["content"] = "\n".join([prose] + fences)


def estimate_tokens(messages):
    """What these messages cost by the fitted rates. See TOKENS_PER_CHAR: only
    the difference between two of these means anything."""
    chars, images = 0, 0
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            chars += len(content)
            continue
        for part in content or []:
            if part.get("type") == "image_url":
                images += 1
            else:
                chars += len(part.get("text") or "")
    return chars * TOKENS_PER_CHAR + images * TOKENS_PER_IMAGE


class ContextMeter:
    """How full the window is, from what the server last measured, so the reply
    budget is never asked for more than the room the prompt leaves."""

    def __init__(self, window):
        self.window = window
        self.prompt, self.estimate = None, 0.0
        # Every call sent, refused ones too.
        self.calls = 0

    def measured(self, prompt_tokens, sent):
        """The server's count for a prompt whose estimate was `sent`."""
        if prompt_tokens is not None:
            self.prompt, self.estimate = prompt_tokens, sent

    def predict(self, messages):
        now = estimate_tokens(messages)
        return now if self.prompt is None else self.prompt + now - self.estimate

    def room(self, messages):
        """Tokens a reply may be given. Can be negative."""
        return int(self.window - self.predict(messages) - CONTEXT_MARGIN)

    def refused(self, asked, messages, error):
        """A context 400. It names the real window, and of the prompt says only
        that prompt + `asked` passed it, so that floor is all that is learned."""
        got = re.search(CONTEXT_ERROR + r" is (\d+)", error)
        if got:
            self.window = int(got.group(1))
        self.measured(max(self.predict(messages), self.window - asked + 1),
                      estimate_tokens(messages))


class ContextExhausted(Exception):
    """The system prompt and the newest exchange alone do not fit. Carries
    CONTEXT_ERROR, which is what files the run as `context_exhausted`."""


def compact_output(o):
    """Replace a tool output with its SUMMARY: block and any error, once.

    Rewriting an old message costs the prefix cache everything after it, so an
    output is rewritten a single time and then left alone. The full text stays
    on disk and `m2c.py recall` brings it back.
    """
    if o.get("compacted") or o.get("discarded"):
        return
    o["compacted"] = True
    kept = [f"[output of command {o['n']} compacted; the full text: "
            f"m2c.py recall {o['n']}]"]
    summary = SUMMARY_BLOCK.findall(o["full"])
    if summary:
        kept.append("SUMMARY:\n" + summary[-1].rstrip())
    if not o["ok"]:
        kept.append(o["full"].strip()[-ERROR_TAIL_CHARS:])
    text = "\n".join(kept)
    if len(text) < len(o["text"]):
        o["message"]["content"][0] = {"type": "text", "text": text}


def compact_recent_outputs(messages, outputs, meter):
    """Compact outputs younger than KEEP_OUTPUTS, oldest first, until the target
    is met -- not all at once, which would discard evidence the target did not
    ask for."""
    target = meter.window * CONTEXT_TARGET
    newest = {id(m) for m in messages[-2:]}
    for o in outputs:
        if meter.predict(messages) <= target:
            break
        if id(o["message"]) not in newest:
            compact_output(o)


def discard_oldest_outputs(messages, outputs, meter):
    """Stub tool outputs from the first command on, until the target is met.

    The assistant message stays, so the model still sees what it ran and in what
    order. Returns the command numbers discarded.
    """
    target = meter.window * CONTEXT_TARGET
    newest = {id(m) for m in messages[-2:]}
    gone = []
    for o in outputs:
        if meter.predict(messages) <= target:
            break
        if id(o["message"]) in newest or o.get("discarded"):
            continue
        o["discarded"] = True
        o["images"] = []
        name = " ".join(o["cmd"].split()[:2])
        o["message"]["content"] = [{"type": "text", "text": (
            f"[output of command {o['n']} ({name} ...) discarded to fit the "
            f"context; m2c.py recall {o['n']}]")}]
        gone.append(o["n"])
    return gone


def remove_oldest_commands(messages, outputs, meter):
    """Take out whole command/output pairs from the oldest on.

    Cuts only up to an assistant message, right after the first user message, so
    the user/assistant alternation stays valid. Returns the numbers removed.
    """
    target = meter.window * CONTEXT_TARGET
    cut = 2
    for i in range(3, len(messages) - 1):
        if messages[i]["role"] == "assistant":
            cut = i
            if meter.predict(messages[:2] + messages[i:]) <= target:
                break
    removed = {id(m) for m in messages[2:cut]}
    gone = [o for o in outputs if id(o["message"]) in removed]
    del messages[2:cut]
    outputs[:] = [o for o in outputs if id(o["message"]) not in removed]
    if gone:
        messages[1]["content"] = (f"{FIRST_MESSAGE}\n\n[commands 1-{gone[-1]['n']} "
                                  f"removed to fit the context]")
    return [o["n"] for o in gone]


def fit_context(messages, outputs, meter):
    """Climb the ladder only as far as the target needs. Returns one trace entry
    per rung that freed something.

    Never touched: the system prompt, the first user message and the newest
    exchange. Lowering max_tokens is not a rung, `ask_model` does it on every
    call.
    """
    rungs = [
        lambda: compact_recent_outputs(messages, outputs, meter),
        lambda: drop_old_images(outputs, 1, 0),
        lambda: discard_oldest_outputs(messages, outputs, meter),
        lambda: remove_oldest_commands(messages, outputs, meter)]
    taken = []
    for rung, step in enumerate(rungs, 1):
        before = meter.predict(messages)
        if before <= meter.window * CONTEXT_TARGET:
            break
        gone = step()
        after = meter.predict(messages)
        if after < before:
            taken.append({"context_retry": rung, "prompt_before": round(before),
                          "prompt_after": round(after)})
            if gone:
                taken[-1]["context_discarded"] = gone
    return taken


def ask_model(client, args, messages, outputs, meter, want, turn, log):
    """The model's reply to `messages`, having made them fit.

    The reply gets `want` tokens or what is left, whichever is less. The ladder
    runs before the call when less is left than an ordinary reply needs, because
    a reply squeezed under that is truncated and wasted, and again after a
    context 400, which only means the prediction was short. Nothing here ends
    the run but ContextExhausted, once there is nothing left to cut.

    Returns the response and the max_tokens it was asked with.
    """
    while True:
        if meter.room(messages) < min(want, args.max_tokens):
            for entry in fit_context(messages, outputs, meter):
                log.append({"turn": turn, "cmd": None, "usage": None,
                            "context_call": meter.calls + 1, **entry})
                print(f"    context: {entry}", flush=True)
        asked = min(want, meter.room(messages))
        if asked < MIN_REPLY_TOKENS:
            raise ContextExhausted(
                f"{CONTEXT_ERROR} {meter.window}: the prompt is about "
                f"{meter.predict(messages):.0f} tokens and nothing is left to cut")
        meter.calls += 1
        sent = estimate_tokens(messages)
        try:
            resp = client.chat.completions.create(
                model=args.model, messages=messages,
                max_tokens=asked, temperature=args.temperature)
        except Exception as exc:                                   # noqa: BLE001
            if CONTEXT_ERROR not in str(exc):
                raise
            # Not `error`: that key is what files a run as context_exhausted.
            log.append({"turn": turn, "cmd": None, "usage": None,
                        "context_call": meter.calls, "context_refused": asked})
            meter.refused(asked, messages, str(exc))
            continue
        usage = read_usage(resp)
        meter.measured(usage and usage["prompt"], sent)
        return resp, asked


# One m2c.py command may take this long before it is stopped.
COMMAND_TIMEOUT_S = 1800


def _text(b):
    if b is None:
        return ""
    return b.decode("utf-8", "replace") if isinstance(b, bytes) else b


def run_command(cmd, python, work, timeout=None):
    """Execute one m2c.py call as an argv, never through a shell.

    A command that runs past the timeout is stopped and comes back to the model
    as an answer -- what it printed so far, and that it was stopped -- not as an
    exception, which would end the whole run (and lose a program already
    written) over one slow instrument on a very large mesh. The rest of the
    budget is the model's to spend; the tool's job is to say what happened.
    """
    timeout = COMMAND_TIMEOUT_S if timeout is None else timeout
    try:
        argv = shlex.split(cmd)
    except ValueError as exc:
        return f"could not parse that command: {exc}", [], False
    if not argv or argv[0] != "m2c.py":
        return ("refused: only m2c.py commands are run here", [], False)
    argv = [python, str(M2C)] + argv[1:]
    try:
        proc = subprocess.run(argv, cwd=work, capture_output=True, text=True,
                              timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        partial = _text(exc.stdout) + _text(exc.stderr)
        out = (partial.rstrip() + "\n\n" if partial.strip() else "") + (
            f"TIMED OUT: `{cmd}` was still running after {timeout:g} s and was "
            f"stopped; the run goes on. "
            + ("Everything it printed before that is above." if partial.strip()
               else "It printed nothing before that.")
            + " It counts as a call. The same command on this mesh will take "
            "as long again. An --emit or --export it was asked for may or may "
            "not have been written; look before building on it.")
        return out, IMAGE_LINE.findall(partial), False
    out = (proc.stdout or "") + (proc.stderr or "")
    images = IMAGE_LINE.findall(out)
    return out, images, proc.returncode == 0


# The verdict word `check: surface` ends a line with when shaped patches cover
# enough of the target to refuse DONE (mesh2cad/surface_gate.py).
NOT_FINISHED = "not finished"


def deliver(out_dir, mesh, python):
    """scripts/pick_best.py's deliver_best on this run, or a record of why it
    could not run. Never ends the run in a traceback: the program on disk is
    still the result, and the record says it was not chosen."""
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "pick_best", ROOT / "scripts" / "pick_best.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.deliver_best(out_dir, mesh, python, M2C)
    except Exception as exc:                                       # noqa: BLE001
        why = (f"the best version could not be chosen ({type(exc).__name__}: "
               f"{exc}); the last write is delivered")
        print(f"deliver: {why}")
        return {"delivered": "last", "why": why}


# Options of the loop itself, forwarded unchanged to every per-part run.
LOOP_OPTIONS = ("base_url", "model", "api_key", "max_calls", "max_tokens",
                "max_tokens_ceiling", "context_window", "keep_outputs",
                "keep_image_commands", "keep_images", "temperature", "python")


def parser():
    ap = argparse.ArgumentParser(
        description="Reconstruct a CadQuery program from a mesh, or from every "
                    "mesh in a directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--mesh", required=True,
                    help="a mesh file (" + ", ".join(MESH_SUFFIXES) + ") or a "
                         "directory of them")
    ap.add_argument("--out", required=True,
                    help="output root; each mesh gets <out>/<mesh stem>/")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1",
                    help="OpenAI-compatible endpoint (e.g. vLLM)")
    ap.add_argument("--model", required=True,
                    help="model name as the endpoint serves it")
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    ap.add_argument("--max-calls", type=int, default=30,
                    help="command budget per attempt (the paper uses 30 for "
                         "DeepCAD and BenchCAD, 60 for the other benchmarks)")
    ap.add_argument("--max-tokens", type=int, default=15000)
    ap.add_argument("--max-tokens-ceiling", type=int, default=30000,
                    help="a truncated reply is retried with more room, up to "
                         "this. Truncation is a limit of the harness, not a "
                         "failure of the model, and nudging a model that never "
                         "finished its sentence just wastes the budget.")
    ap.add_argument("--context-window", type=int, default=CONTEXT_WINDOW,
                    help="the server's context length, in tokens. A context "
                         "error from the server corrects it.")
    ap.add_argument("--keep-outputs", type=int, default=KEEP_OUTPUTS,
                    help="how many commands an output stays whole for")
    ap.add_argument("--keep-image-commands", type=int, default=KEEP_IMAGE_COMMANDS,
                    help="how many image-printing commands keep their renders")
    ap.add_argument("--keep-images", type=int, default=KEEP_IMAGES,
                    help="how many renders the conversation may hold in all")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--python", default=sys.executable,
                    help="interpreter the m2c.py instruments run under")
    batch = ap.add_argument_group("batch")
    batch.add_argument("--attempts", type=int, default=1,
                       help="retry a part until an attempt finishes (empty "
                            "residual), at most this many times; the attempt "
                            "with the smallest residual is kept")
    batch.add_argument("--jobs", type=int, default=1,
                       help="parts reconstructed in parallel")
    batch.add_argument("--skip-existing", action="store_true",
                       help="skip parts whose metrics.json already exists")
    batch.add_argument("--no-export", action="store_true",
                       help="do not export the delivered program to STL")
    # Internal: run the bare loop on one mesh, with --out as the run directory.
    # This is what every per-part subprocess is.
    ap.add_argument("--loop", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--workdir", default=None, help=argparse.SUPPRESS)
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    if args.loop:
        sys.exit(reconstruct(args))
    sys.exit(run_batch(args))


def reconstruct(args):
    """The agent loop on one mesh, writing into the run directory `args.out`.

    `--workdir` is where commands run, and the mesh is referred to relative to
    it: absolute paths are long, and a model abbreviated one to
    '/.../part.stl', which is not a file. Short paths are not cosmetic.

    Returns 0 when the run finished (empty residual), 3 otherwise.
    """

    from openai import OpenAI
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Every output whole, for `m2c.py recall`, which runs in the same environment.
    saved_outputs = (out_dir / "outputs").resolve()
    saved_outputs.mkdir(exist_ok=True)
    os.environ["M2C_OUTPUTS"] = str(saved_outputs)
    work = Path(args.workdir or out_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    # What the model is shown and types: short, relative to `work`.
    def rel(p):
        p = Path(p).resolve()
        try:
            return str(p.relative_to(work))
        except ValueError:
            return str(p)
    mesh_ref = rel(args.mesh)
    out_ref = rel(out_dir)
    emit = out_dir / "reconstruction.py"
    emit_ref = rel(emit)

    guide = (ROOT / "prompts" / "reconstruction.md").read_text()
    tools = (ROOT / "prompts" / "instruments.md").read_text()

    system = (
        guide + "\n\n" + tools + "\n\n"
        "## How this session works\n\n"
        "**You can see the renders.** When an instrument prints an `IMAGE:`\n"
        "line, that PNG is attached to this conversation and you are looking at\n"
        "it. Read the picture for topology -- how many bodies, whether a wall is\n"
        "a wall or a pair of strips, where a feature sits -- and read the\n"
        "printed numbers for dimensions. A model that assumed it was blind here\n"
        "spent 82% of a run's output tokens re-deriving from coordinate lists\n"
        "what one overview render showed outright, and never finished the part.\n\n"
        f"**Old outputs shrink.** An output stays whole for {args.keep_outputs}\n"
        "commands. After that only its `SUMMARY:` block stays here. Nothing is\n"
        "lost: `m2c.py recall N` prints command N's output again, N being the\n"
        "number on its `[command N/...]` line, and `--grep TEXT`, `--ring R2` or\n"
        "`--lines a:b` print a part of it. A recall costs a command.\n\n"
        f"The mesh is at `{mesh_ref}`. Write every render under `{out_ref}`.\n"
        f"Build the program with `--emit {emit_ref}`, and pass `--base "
        f"{emit_ref}` on\n"
        "every feature after the first so they accumulate.\n\n"
        "## Reply format\n\n"
        "Put the COMMAND FIRST, before any reasoning:\n\n"
        "    ```bash\n"
        "    m2c.py section part.stl --plane XY --offset B2\n"
        "    ```\n"
        "    then, after it, why you chose that and what in the last output\n"
        "    made you choose it.\n\n"
        "The command goes first because a long reply can be cut off before it\n"
        "ends, and a turn whose command was cut off is a turn wasted -- that\n"
        "cost a whole reconstruction here.\n\n"
        "Keep the reasoning SHORT and put your conclusion at the end of it.\n"
        "After two more turns, only the last few hundred characters of this\n"
        "reply survive in the conversation -- the argument in the middle is\n"
        "dropped and the turn is marked as compacted. This is not a punishment:\n"
        "it is what makes a long reconstruction possible at all. Deliberation\n"
        "filled the window five times faster than evidence did, and runs were\n"
        "dying of it with the part half built. So write the conclusion you want\n"
        "your later self to read, and write it last.\n\n"
        "One command per turn; nothing else is executed. When the residual is\n"
        "empty and the part is complete, reply DONE instead of a command.\n\n"
        f"You have at most {args.max_calls} commands."
    )

    messages = [{"role": "system", "content": system},
                {"role": "user", "content": FIRST_MESSAGE}]

    log, calls, nudges, t0 = [], 0, 0, time.time()
    tokens = TokenLedger()
    last_residual, last_residual_clean = None, False
    # Every residual this run has measured on a program it actually
    # wrote, in order. Three identical readings mean the last two
    # features changed nothing, which is the evidence a concession
    # needs and the model cannot fake by asserting it.
    residual_history = []
    # Commands that have already been rejected once. Re-sending one
    # verbatim is the signature of a stuck model rather than a retry.
    failed_before = set()
    # The last --check that saved nothing, if the model has not
    # emitted since. A run is given one last turn to write it.
    pending_preview = None
    last_call_used = False
    # How many times DONE has been said and refused since the last emit. The
    # refusal text asks the model to "say DONE again and say what is left", so a
    # second DONE must not be refused on the same test as the first.
    dones_refused = 0
    # The `check: surface` line of the last emitted check, and whether a
    # DONE has already been refused on it. Refused once per emitted version:
    # a shape the model has been told about and still cannot build is a
    # concession, not a reason to spend its remaining calls saying no.
    last_surface, surface_refused = None, False
    # The last command and the output it produced. A command re-sent verbatim
    # and answered identically has not advanced anything, however successfully
    # it ran, and must not count as progress.
    last_cmd_out = (None, None)
    budget = args.max_tokens
    meter = ContextMeter(args.context_window)
    # Every tool output in the conversation, for the ladder: which message holds
    # it, the command that made it, and its text as first sent.
    outputs = []
    while calls < args.max_calls:
        # Before the call, not after it. A turn that only reasons never reaches
        # the bottom of the loop, and those are the turns that bloat hardest --
        # so trimming there let exactly the wrong conversations grow unchecked.
        compact_old_reasoning(messages, KEEP_REASONING_FOR_TURNS)
        for o in outputs[:-args.keep_outputs or None]:
            compact_output(o)
        try:
            resp, asked = ask_model(client, args, messages, outputs, meter,
                                    budget, calls, log)
        except Exception as exc:                                   # noqa: BLE001
            # A call that raised brings back no usage, whatever the server may
            # have done with the prompt. There is nothing to count here and
            # nothing worth guessing.
            log.append({"error": f"{type(exc).__name__}: {exc}", "usage": None})
            print(f"model call failed: {exc}", flush=True)
            break
        usage = tokens.add(resp)
        say = resp.choices[0].message.content or ""
        why_stopped = resp.choices[0].finish_reason
        messages.append({"role": "assistant", "content": say})

        cmd, superseded = extract_command(say)
        if cmd is None:
            # A reply that was CUT OFF is not a reply that said anything, and it
            # is tested first. The word DONE turns up constantly mid-thought --
            # "I am not done", "once the boss is done" -- so a truncated sentence
            # containing it must not be read as a completion claim. Truncation
            # instead reaches the doubled-budget retry below, which exists for
            # exactly this case.
            #
            # As a whole word, too. "DONE" is a substring of ABANDONED.
            #
            # ...unless the retry that guard exists to reach cannot happen. The
            # budget only doubles while it is below the ceiling; at the ceiling a
            # truncated reply would fall straight through to a nudge and its DONE
            # would be lost, even with a residual under the concession gate already
            # written to disk. A false DONE with nothing built is caught on its own
            # two lines below.
            at_ceiling = budget >= args.max_tokens_ceiling or asked < budget
            readable = (why_stopped != "length"
                        or (at_ceiling and last_residual is not None))
            if readable and re.search(r"\bDONE\b", say.upper()):
                # DONE has to be backed by evidence: a model can declare a part
                # finished after one feature without ever asking what was left. This
                # does not tell it what to build -- only that something is still
                # missing and it has not looked.
                if last_residual is None:
                    messages.append({"role": "user", "content":
                                     "Nothing has been built and checked yet, "
                                     "so there is no evidence the part is "
                                     "complete. Build a feature with "
                                     "`--emit ... --check`."})
                    nudges += 1
                    log.append({"turn": calls, "say": say, "cmd": None,
                                "done_refused": "nothing built",
                                "usage": usage})
                    if nudges > MAX_NUDGES:
                        log.append({"turn": calls, "cmd": None, "gave_up": True,
                                    "after": "refused DONE, nothing built"})
                        break
                    continue
                if (last_surface and NOT_FINISHED in last_surface
                        and not surface_refused):
                    # The volume says done (or nearly), the target's surface does not:
                    # thin features can pass the volume test while a large share of the
                    # target's triangles are still off.
                    surface_refused = True
                    messages.append({"role": "user", "content":
                                     "Not finished: the volume is nearly all "
                                     "there, but the target's surface is not:\n  "
                                     + last_surface + "\nThat is a shape the "
                                     "residual cannot weigh. Look at it: `section "
                                     "<mesh> --plane P --offset <a level through "
                                     "the largest patch> --built PROG --fill`, or "
                                     "`residual --plane P`. Build it, or say DONE "
                                     "again and say in one line why it cannot be "
                                     "built."})
                    nudges += 1
                    log.append({"turn": calls, "say": say, "cmd": None,
                                "done_refused": last_surface, "usage": usage})
                    continue
                if not last_residual_clean:
                    # A part the instruments cannot finish can never produce a clean
                    # residual, so refusing DONE until it is clean would nudge the model
                    # until MAX_NUDGES turned a correct concession into `gave_up`.
                    # Refusing once is right -- a model can declare a part finished far
                    # too early -- but refusing a model that has stopped making progress
                    # just spends its remaining calls saying no.
                    #
                    # So: refused while the residual is still falling, accepted
                    # as a CONCESSION once it has stopped. That is not a claim
                    # the part is right. `complete` stays false, the exit code
                    # still asks for a retry, and the trace records what was
                    # left rather than the fact that we ran out of patience.
                    # The refusal below ends "say DONE again and say in one
                    # line what is left" -- so a second DONE answers a question
                    # the harness asked, and refusing it on the same test that
                    # refused the first makes the request a formality.
                    #
                    # Honoured only when the residual has stopped falling,
                    # which is the narrower half of what `stalled` tests: that
                    # one needs three readings before it can fire, so a run
                    # that emitted twice, got stuck, and said DONE twice would be
                    # refused for want of history rather than for want of
                    # progress. A model quitting while the number is STILL
                    # coming down is a different case and is still pushed --
                    # it has not run out of things to try, it has run out of
                    # patience, and the two should not share a label.
                    dones_refused += 1
                    stuck = (stalled(residual_history)
                             or near_enough(last_residual)
                             or (dones_refused > 1
                                 and not improving(residual_history)))
                    if stuck:
                        log.append({"turn": calls, "say": say, "cmd": None,
                                    "conceded": last_residual,
                                    "unchanged_checks": len(residual_history),
                                    "usage": usage})
                        print(f"    conceded with {last_residual}", flush=True)
                        break
                    messages.append({"role": "user", "content":
                                     "Not finished: the last check said\n  "
                                     + last_residual +
                                     "\nRun `residual` to see where those "
                                     "bodies are, then build them. If you have "
                                     "already tried and the number is not "
                                     "moving, say DONE again and say in one "
                                     "line what is left and why you cannot "
                                     "reach it."})
                    nudges += 1
                    log.append({"turn": calls, "say": say, "cmd": None,
                                "done_refused": last_residual,
                                "usage": usage})
                    if nudges > MAX_NUDGES:
                        log.append({"turn": calls, "cmd": None, "gave_up": True,
                                    "after": "refused DONE while it kept asking"})
                        break
                    continue
                log.append({"turn": calls, "say": say, "cmd": None,
                            "done": True, "usage": usage})
                break
            # A turn that only reasons is not a tool call and is not charged
            # as one. Charging it halved the effective budget: the model
            # narrated the last output, got nudged, then acted -- two turns per
            # action, so ten calls bought five commands. `nudges` still bounds
            # the loop so a model that never emits a command cannot spin.
            # Truncation is the likeliest reason a command is missing, and it
            # is not the model refusing to act -- it never got to the end of
            # its sentence. Say which it was, and give it the whole budget back
            # rather than nudging it into the same wall.
            cut = why_stopped == "length"
            if cut and not at_ceiling:
                # Give it room and ask again, rather than telling a model that
                # never reached the end of its sentence to try harder: the
                # reasoning simply did not fit.
                budget = min(budget * 2, args.max_tokens_ceiling)
                messages.pop()          # drop the truncated half-reply
                # The reply is dropped from the conversation; its tokens are
                # not dropped from the ledger. They were generated.
                log.append({"turn": calls, "cmd": None,
                            "truncated_retry": budget, "usage": usage})
                print(f"    reply truncated -- retrying with {budget} tokens",
                      flush=True)
                continue
            nudges += 1
            if nudges > MAX_NUDGES:
                log.append({"turn": calls, "say": say, "cmd": None,
                            "gave_up": True, "usage": usage})
                break
            messages.append({"role": "user", "content":
                             ("Your reply was CUT OFF before the command. Give "
                              "the command first this time, in a ```bash "
                              "fence, and keep any reasoning to two sentences."
                              if cut else
                              "That turn had no command. Every reply must end "
                              "with one ```bash fence containing a single "
                              "m2c.py command -- reasoning first is fine, but "
                              "the command must be there. Or reply DONE.")})
            log.append({"turn": calls, "say": say, "cmd": None,
                        "finish_reason": why_stopped, "usage": usage})
            continue

        print(f"[{calls + 1}/{args.max_calls}] {cmd[:120]}", flush=True)
        out, images, ok = run_command(cmd, args.python, work)
        repeated = cmd in failed_before
        if not ok:
            failed_before.add(cmd)
        calls += 1
        (saved_outputs / f"{calls:03d}.txt").write_text(out)
        # A command that ran, and answered exactly what it answered last turn,
        # has moved nothing; `ok` alone would let a no-op loop reset the counter
        # forever. Comparing the OUTPUT and not just the command on purpose --
        # re-running `section` at a new level is a different question with the
        # same words, and only the answer can tell the two apart.
        no_op = ok and (cmd, out) == last_cmd_out
        last_cmd_out = (cmd, out)
        # Reset only when the command actually RAN. A rejected command is not
        # progress, and resetting on it would make MAX_NUDGES unreachable for a
        # model that keeps re-sending commands argparse rejects.
        if ok and not no_op:
            nudges = 0
            dones_refused = 0
        elif no_op:
            nudges += 1
            print("    identical command, identical output -- nothing moved",
                  flush=True)
        else:
            nudges += 1
            if repeated:
                # Verbatim repetition of a command that just failed is not a
                # second attempt at anything.
                nudges += 1
        log.append({"turn": calls, "say": say, "cmd": cmd, "ok": ok,
                    "finish_reason": why_stopped, "usage": usage,
                    "superseded": superseded, "no_op": no_op,
                    "out": elide(out), "images": images})
        # The same bound the no-command path has always had. It existed only
        # there, so a model that kept sending commands argparse rejected was
        # never stopped by it -- the counter it was tripping was checked on a
        # branch it never reached.
        if nudges > MAX_NUDGES:
            log.append({"turn": calls, "cmd": None, "gave_up": True,
                        "after": "repeating itself" if no_op
                                 else "rejected commands"})
            print(f"    {nudges} turns without progress -- stopping", flush=True)
            break

        for ln in out.splitlines():
            if ln.startswith("check: residual"):
                line = ln.split("check: ", 1)[1].strip()
                # The number carries whether it describes a file: without the
                # label, trace.json would record a residual for a program that
                # was never written, and pick_best and every summary read
                # exactly that field. Saying which it was costs eight words.
                last_residual = (line if "--emit" in cmd else
                                 line + "  [PREVIEW: this command had no "
                                        "--emit, so it describes a program "
                                        "that was not written]")
                if "--emit" in cmd:
                    residual_history.append(residual_total(line))
                    pending_preview = None
                else:
                    # The command that measured a program nobody saved. Kept so
                    # the run cannot simply end here having previewed a part it
                    # never wrote.
                    pending_preview = cmd
                # A clean residual only counts as evidence about the program on
                # disk if THIS command wrote that program. `--base X --check`
                # without `--emit` measures base-plus-the-new-feature and saves
                # nothing, so the number describes a program that does not
                # exist, while reconstruction.py may still hold something
                # much worse. The stopping rule must read the program on disk.
                clean = "nothing above the sliver floor" in ln
                last_residual_clean = clean and "--emit" in cmd
                if "--emit" in cmd:
                    last_surface, surface_refused = None, False
                if clean and "--emit" not in cmd:
                    messages.append({"role": "user", "content":
                                     "That check passed, but it had no --emit, "
                                     "so nothing was written -- "
                                     f"{emit_ref} is still whatever the last "
                                     "--emit left there. Re-run that same "
                                     f"command with --emit {emit_ref} before "
                                     "you call the part finished."})
            if ln.startswith("check: surface") and "--emit" in cmd:
                last_surface = ln.split("check: ", 1)[1].strip()
        # Which command actually ran, when the reply held more than one. The
        # model drafts and discards commands as it reasons; it is entitled to
        # know that the harness took the last and not one of the drafts.
        preface = "" if ok else "THE COMMAND FAILED.\n"
        if repeated:
            preface += ("[you have already sent this exact command and it was "
                        "rejected the same way. Read the error and change the "
                        "command, or run a different instrument -- re-sending "
                        "it will not produce a different answer.]\n")
        if superseded:
            preface += (f"[this reply contained {len(superseded) + 1} commands; "
                        f"the LAST one ran: {cmd}]\n")
        content = [{"type": "text", "text": preface + elide(out)}]
        messages.append({"role": "user", "content": content})
        outputs.append({"n": calls, "cmd": cmd, "text": content[0]["text"],
                        "full": out, "ok": ok,
                        "message": messages[-1], "images": []})
        attach_images(outputs[-1], images, outputs)
        drop_old_images(outputs, args.keep_image_commands, args.keep_images)

    # One last turn to write down what was measured.
    #
    # A run must not end having checked a program it never saved. `--base X
    # --check` with no `--emit` measures base-plus-the-new-feature and writes
    # nothing, so the run would stop holding a number for a part that exists only
    # in the output of a command.
    #
    # This asks once, off the call budget, and names the command to re-run. It
    # cannot loop: `last_call_used` allows exactly one, and a model that answers
    # with anything else gets no second offer.
    if pending_preview and not last_call_used:
        last_call_used = True
        print(f"    last call: {pending_preview} was never written", flush=True)
        messages.append({"role": "user", "content":
                         "This run is about to end and your last check had no "
                         f"--emit, so nothing it measured was saved. "
                         f"{emit_ref} still holds whatever your last --emit "
                         "left there. Re-send exactly this command with "
                         f"`--emit {emit_ref}` added:\n\n    "
                         f"{pending_preview}\n\nReply with that one command and "
                         "nothing else."})
        try:
            resp, _asked = ask_model(client, args, messages, outputs, meter,
                                     args.max_tokens, calls, log)
            usage = tokens.add(resp)
            say = resp.choices[0].message.content or ""
            cmd, _superseded = extract_command(say)
        except Exception as exc:                                   # noqa: BLE001
            log.append({"error": f"{type(exc).__name__}: {exc}", "usage": None})
            cmd = None
        if cmd and "--emit" in cmd:
            out, images, ok = run_command(cmd, args.python, work)
            calls += 1
            log.append({"turn": calls, "say": say, "cmd": cmd, "ok": ok,
                        "last_call": True, "usage": usage,
                        "out": elide(out), "images": images})
            for ln in out.splitlines():
                if ln.startswith("check: residual"):
                    last_residual = ln.split("check: ", 1)[1].strip()
                    last_residual_clean = ("nothing above the sliver floor" in ln)
            if ok:
                pending_preview = None
        else:
            log.append({"turn": calls, "say": say, "cmd": cmd,
                        "last_call": True, "refused": "no --emit in the reply",
                        "usage": usage})

    # Whether the run FINISHED, by the only evidence available without the
    # ground truth: an empty residual. The caller can retry on it.
    complete = bool(emit.exists() and last_residual_clean)
    # Deliver the best valid version the run saved, not merely the last one
    # written: a broken last write must not replace a valid earlier version.
    # A finished run is never changed: its clean last version is the best there
    # is. The model never sees this.
    delivered = deliver(out_dir, args.mesh, args.python)
    # HOW it ended, not just whether it finished. These were indistinguishable
    # before, and they are four different results: a part that is right, a
    # model that stopped on purpose with the residual flat, a model that ran out
    # of patience being told no, and a budget that ran out mid-build. Half of
    # all endings were the middle two and both were filed as "gave up".
    if complete:
        ending = "complete"
    elif any("conceded" in e for e in log):
        ending = "conceded"
    elif pending_preview:
        ending = "ended_on_preview"
    elif any(e.get("gave_up") for e in log):
        ending = "nudges_exhausted"
    elif any("maximum context length" in str(e.get("error", "")) for e in log):
        ending = "context_exhausted"
    elif calls >= args.max_calls:
        ending = "call_budget_spent"
    else:
        ending = "stopped"
    (out_dir / "trace.json").write_text(json.dumps({
        "mesh": args.mesh, "model": args.model, "calls": calls,
        "seconds": round(time.time() - t0, 1),
        "tokens": tokens.totals(),
        "complete": complete, "ending": ending,
        "last_residual": last_residual,
        "delivered": delivered,
        "log": log}, indent=1))
    # Tokens first: the batch runner keeps only the last line of a run, and the
    # line worth keeping is the one that says whether the part is finished.
    print(tokens.summary())
    print(f"{calls} call(s), {time.time() - t0:.0f}s, "
          f"program {'written' if emit.exists() else 'NOT written'}, {ending}"
          f"{'' if complete else ': ' + (last_residual or 'no residual seen')}")
    # Non-zero when the part is not finished, so a wrapper can try again. These
    # runs are stochastic, and a retry is cheap compared with the spread between
    # attempts on a hard part.
    return 0 if complete else 3



# --- batch: one mesh or a directory of them ----------------------------------

def find_meshes(path):
    """The meshes to reconstruct: the file itself, or every mesh file directly
    inside the directory, sorted by name."""
    path = Path(path)
    if path.is_file():
        return [path]
    if path.is_dir():
        found = sorted(p for p in path.iterdir()
                       if p.is_file() and p.suffix.lower() in MESH_SUFFIXES)
        if not found:
            raise SystemExit(f"agent_run: no {'/'.join(MESH_SUFFIXES)} files in {path}")
        return found
    raise SystemExit(f"agent_run: {path} is neither a mesh file nor a directory")


def stage_mesh(mesh, dest):
    """Put the input where the agent expects it, as STL."""
    if mesh.suffix.lower() == ".stl":
        shutil.copyfile(mesh, dest)
        return
    import trimesh
    trimesh.load(mesh, force="mesh").export(dest)


def _loop_argv(args):
    argv = []
    for name in LOOP_OPTIONS:
        argv += ["--" + name.replace("_", "-"), str(getattr(args, name))]
    return argv


def _run_attempt(args, work, log, echo):
    """One attempt in its own process: work/run is its run directory, and the
    agent sees the mesh as `part.stl`, relative to work/. Returns the exit code."""
    cmd = [sys.executable, str(Path(__file__).resolve()), "--loop",
           "--mesh", "part.stl", "--out", "run", "--workdir", "."] + _loop_argv(args)
    env = dict(os.environ, PYVISTA_OFF_SCREEN="true")
    with subprocess.Popen(cmd, cwd=work, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, env=env) as proc:
        for line in proc.stdout:
            log.write(line)
            log.flush()
            if echo:
                print(line, end="", flush=True)
    return proc.returncode


def _pick_attempt(work, python):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "pick_best", ROOT / "scripts" / "pick_best.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.pick_attempt(work, python, M2C)


def run_part(args, mesh, out_root, echo):
    """Reconstruct one mesh into out_root/<stem>/ and export what was delivered."""
    stem = mesh.stem
    work = (out_root / stem).resolve()
    if args.skip_existing and (work / "metrics.json").exists():
        return {"part": stem, "status": "skipped"}
    work.mkdir(parents=True, exist_ok=True)
    for old in [work / "run", *work.glob("attempt_*")]:
        shutil.rmtree(old, ignore_errors=True)
    for old in ("metrics.json", f"{stem}.py", f"{stem}.stl"):
        (work / old).unlink(missing_ok=True)
    stage_mesh(mesh, work / "part.stl")

    t0 = time.time()
    rc, used = None, 0
    with open(work / "agent.log", "w") as log:
        for a in range(1, args.attempts + 1):
            used = a
            shutil.rmtree(work / "run", ignore_errors=True)
            log.write(f"=== attempt {a} of {args.attempts}\n")
            rc = _run_attempt(args, work, log, echo)
            if rc == 0:
                break
            if a < args.attempts and (work / "run").exists():
                shutil.copytree(work / "run", work / f"attempt_{a}")
        if args.attempts > 1:
            with contextlib.redirect_stdout(log):
                try:
                    _pick_attempt(work, args.python)
                except Exception as exc:                           # noqa: BLE001
                    print(f"pick_best failed: {type(exc).__name__}: {exc}")
    secs = time.time() - t0
    status = {0: "done", 3: "incomplete"}.get(rc, f"crashed({rc})")

    trace = {}
    try:
        trace = json.loads((work / "run" / "trace.json").read_text())
    except Exception:                                              # noqa: BLE001
        pass
    prog = work / "run" / "reconstruction.py"
    exported = False
    if prog.exists() and prog.stat().st_size:
        shutil.copyfile(prog, work / f"{stem}.py")
        if not args.no_export:
            with open(work / "export.log", "w") as elog:
                r = subprocess.run(
                    [args.python, str(ROOT / "scripts" / "export_stl.py"), str(prog),
                     str(work / f"{stem}.stl"), "--target", str(work / "part.stl")],
                    cwd=work, stdout=elog, stderr=subprocess.STDOUT,
                    env=dict(os.environ, PYVISTA_OFF_SCREEN="true"))
            exported = r.returncode == 0 and (work / f"{stem}.stl").exists()
    metrics = {
        "part": stem, "mesh": str(mesh), "status": status, "attempts": used,
        "calls": trace.get("calls"), "ending": trace.get("ending"),
        "complete": trace.get("complete"), "last_residual": trace.get("last_residual"),
        "tokens": trace.get("tokens"), "time_spent_min": round(secs / 60, 1),
        "program": prog.exists(), "exported": exported, "model": args.model,
    }
    (work / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def run_batch(args):
    meshes = find_meshes(args.mesh)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    stems = [m.stem for m in meshes]
    dup = sorted({s for s in stems if stems.count(s) > 1})
    if dup:
        raise SystemExit(f"agent_run: several meshes share a stem: {', '.join(dup)}")
    print(f"{len(meshes)} mesh(es), model {args.model} at {args.base_url}, "
          f"{args.jobs} job(s) -> {out_root}", flush=True)
    echo = args.jobs == 1 and len(meshes) == 1

    def one(mesh):
        m = run_part(args, mesh, out_root, echo)
        print(f"{m['part']} :: {m['status']}"
              + (f", {m['calls']} calls, {m['time_spent_min']} min, {m['ending']}"
                 if m["status"] != "skipped" else ""), flush=True)
        return m

    if args.jobs > 1:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            results = list(pool.map(one, meshes))
    else:
        results = [one(m) for m in meshes]
    done = sum(1 for m in results if m["status"] == "done")
    print(f"{done}/{len(results)} finished with an empty residual; "
          f"`python evaluate.py --predictions {out_root}` scores them")
    return 0 if all(m["status"] in ("done", "skipped") for m in results) else 3


if __name__ == "__main__":
    main()
