"""Flag sentences a prompt rewrite left broken. `python -m mesh2cad.prompt_check`.

A prompt is read by a model that cannot ask what a cut sentence meant, and a
broken paragraph reads as fine in a diff. Only paragraph ENDS are looked at: a
line wrapped inside a paragraph ends anywhere, a paragraph does not.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROMPTS = sorted((ROOT / "prompts").glob("*.md"))
GENERATED = {"instruments.md"}
# A paragraph cannot end on one of these.
DANGLING = {"a", "an", "and", "as", "at", "because", "but", "by", "for", "from", "if",
            "in", "into", "is", "it", "its", "of", "on", "or", "so", "than", "that",
            "the", "their", "then", "these", "this", "to", "was", "were", "when",
            "which", "while", "with", "without"}
ENDS = (".", ":", "!", "?", ")", "`", "*", "]", '"', "'", ";", "--", ">", "|")
BLOCK = re.compile(r"\s*(#|\||```|    |\t|<!--)")
# Wrapped text stays under this. A line short of it by more than SLACK with room
# for the next word is a line someone broke by hand, usually around an insertion.
WIDTH = 86
SLACK = 16


def paragraphs(lines):
    """(first line number, its lines) of each prose paragraph."""
    para, start, fenced = [], 0, False
    for n, line in enumerate([*lines, ""], 1):
        if line.lstrip().startswith("```"):
            fenced = not fenced
        if not line.strip() or fenced or BLOCK.match(line):
            if para:
                yield start, para
            para = []
            continue
        if not para:
            start = n
        para.append(line.strip())


def problems(text, wrapped=True):
    """(line, why, the text) for each place that looks cut or badly wrapped.
    `wrapped` is False for generated text, whose lines are as long as they are."""
    lines = text.splitlines()
    found = []
    for n, para in paragraphs(lines):
        last = para[-1]
        if wrapped and para[0][0].islower() and len(para) > 1:
            found.append((n, "starts in mid-sentence", para[0][:60]))
        if last.endswith(ENDS):
            continue
        word = re.sub(r"\W+$", "", last).rsplit(None, 1)[-1].lower()
        # A lone line is a generated one-line description, not a sentence.
        if word in DANGLING or len(para) > 1:
            found.append((n + len(para) - 1, f"cut off after {word!r}", last[-60:]))
    fenced = False
    for n, line in enumerate(lines if wrapped else [], 1):
        if line.lstrip().startswith("```"):
            fenced = not fenced
        indent = len(line) - len(line.lstrip())
        # Four spaces is a literal block: a table or a synopsis, laid out by hand.
        if fenced or indent == 4 or line.lstrip().startswith(("|", "#")):
            continue
        if len(line) > WIDTH:
            found.append((n, f"{len(line)} characters on one line", line[:60]))
        following = lines[n].strip() if n < len(lines) else ""
        if (line.strip() and following and following[0].isalnum()
                and not line.rstrip().endswith((":", ".", ";", "?"))
                and len(lines[n]) - len(following) == indent
                and len(line) + 1 + len(following.split()[0]) < WIDTH - SLACK):
            found.append((n, "wrapped short of the next word", line.strip()[-60:]))
    return found


def main():
    bad = 0
    for path in PROMPTS:
        for n, why, tail in problems(path.read_text(), path.name not in GENERATED):
            print(f"{path.relative_to(ROOT)}:{n}: {why}: ...{tail}")
            bad += 1
    if bad:
        sys.exit(f"{bad} paragraph(s) look cut off")


if __name__ == "__main__":
    main()
