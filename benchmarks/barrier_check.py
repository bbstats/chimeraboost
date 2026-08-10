"""Check a proposed idea against the barriers this project has already paid for.

`benchmarks/BARRIERS.md` is the source of truth; this script only matches an idea
description against its tags and prints what the idea owes an argument about. It
is deliberately dumb -- keyword matching, no model -- because its value is that it
gets RUN before a `--decide` run is spent, not that it is clever.

A match is not a veto and no match is not a clearance: a barrier says "this
family has been closed before, say why yours is different, in the plan file,
before the run."

Usage:
  python benchmarks/barrier_check.py "cheapen the audition on small data"
  python benchmarks/barrier_check.py --list          # every barrier, one line each
  python benchmarks/barrier_check.py --show B2       # one barrier in full

Exit code is 0 always -- this reports, it does not gate.
"""
import argparse
import os
import re
import sys

BARRIERS_MD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "BARRIERS.md")

_HEAD = re.compile(r"^###\s+(B\d+)\s+[-—]+\s+(.*)$")
_TAGS = re.compile(r"^tags:\s*(.*)$", re.IGNORECASE)


def _words(text):
    """Lowercase word list, with '-' and '_' treated as spaces."""
    return re.findall(r"[a-z0-9]+", re.sub(r"[-_]", " ", text.lower()))


def parse_barriers(path=BARRIERS_MD):
    """Return [{id, title, tags, body}] parsed from BARRIERS.md."""
    with open(path, encoding="utf-8") as fh:
        lines = fh.read().splitlines()

    out, cur = [], None
    for line in lines:
        head = _HEAD.match(line)
        if head:
            cur = {"id": head.group(1), "title": head.group(2).strip(),
                   "tags": [], "body": []}
            out.append(cur)
            continue
        if cur is None:
            continue
        tags = _TAGS.match(line.strip())
        if tags and not cur["tags"]:
            cur["tags"] = [t.strip() for t in tags.group(1).split(",") if t.strip()]
            continue
        cur["body"].append(line)

    for b in out:
        # Drop leading/trailing blank lines; keep the prose as written.
        body = b["body"]
        while body and not body[0].strip():
            body.pop(0)
        while body and not body[-1].strip():
            body.pop()
        b["body"] = "\n".join(body)
    return out


def _token_hit(tag_word, idea_word):
    """A tag word matches an idea word on a shared prefix of >= 5 chars.

    So 'shrink' finds the 'shrinkage' barrier and 'auditions' finds 'audition',
    while 'cat' does not match 'catch' and 'leaf' does not match 'leaftune'.
    """
    if tag_word == idea_word:
        return True
    if len(tag_word) < 5 or len(idea_word) < 5:
        return False
    return tag_word.startswith(idea_word) or idea_word.startswith(tag_word)


def match(idea, barriers):
    """Return [(barrier, [matched tags])], most-matched first."""
    idea_words = _words(idea)
    hits = []
    for b in barriers:
        matched = []
        for tag in b["tags"]:
            tw = _words(tag)
            if not tw:
                continue
            # Multi-word tags ('small-data') must appear as a contiguous run.
            for i in range(len(idea_words) - len(tw) + 1):
                if all(_token_hit(a, c) for a, c in zip(tw, idea_words[i:i + len(tw)])):
                    matched.append(tag)
                    break
        if matched:
            hits.append((b, matched))
    hits.sort(key=lambda h: (-len(h[1]), int(h[0]["id"][1:])))
    return hits


def _ascii(text):
    """Windows consoles here are cp1252; keep printed prose readable."""
    for src, dst in (("—", "--"), ("–", "-"), ("’", "'"), ("−", "-"),
                     ("“", '"'), ("”", '"'), ("×", "x"), ("≤", "<="),
                     ("≥", ">="), ("~", "~")):
        text = text.replace(src, dst)
    return text.encode("ascii", "replace").decode("ascii")


def first_sentence(body, limit=110):
    """The barrier's opening claim, reflowed onto one line."""
    para = []
    for line in body.splitlines():
        if not line.strip():
            break
        para.append(line.strip())
    text = _ascii(" ".join(para))
    cut = text.find(". ")
    if cut != -1:
        text = text[:cut + 1]
    if len(text) > limit:
        text = text[:limit - 3].rstrip() + "..."
    return text


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("idea", nargs="*", help="the proposed idea, in a sentence")
    ap.add_argument("--list", action="store_true", help="every barrier, one line each")
    ap.add_argument("--show", metavar="ID", help="print one barrier in full (e.g. B2)")
    args = ap.parse_args(argv)

    barriers = parse_barriers()
    if not barriers:
        print(f"no barriers parsed from {BARRIERS_MD} -- check its heading format")
        return 0

    if args.list:
        for b in barriers:
            print(f"{b['id']:>4}  {_ascii(b['title'])}")
        return 0

    if args.show:
        want = args.show.strip().upper()
        for b in barriers:
            if b["id"] == want:
                print(f"### {b['id']} - {_ascii(b['title'])}\n")
                print(_ascii(b["body"]))
                return 0
        print(f"no barrier {want}; try --list")
        return 0

    idea = " ".join(args.idea).strip()
    if not idea:
        ap.print_help()
        return 0

    hits = match(idea, barriers)
    print(f'idea: "{idea}"')
    print(f"{len(barriers)} barriers on file, {len(hits)} matched\n")

    if not hits:
        print("No barrier matched. That is NOT a clearance -- it means the")
        print("keywords did not land. Read BARRIERS.md yourself before the run,")
        print("and add tags here if this idea's family is already closed.")
        return 0

    for b, matched in hits:
        print(f"{b['id']} - {_ascii(b['title'])}")
        print(f"     matched on: {', '.join(matched)}")
        first = first_sentence(b["body"])
        if first:
            print(f"     {first}")
        print(f"     full text: python benchmarks/barrier_check.py --show {b['id']}")
        print()

    print("Each of these owes an argument in the plan file, written BEFORE the")
    print("run: which finding is wrong, or why it does not apply here.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
