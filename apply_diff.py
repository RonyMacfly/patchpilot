#!/usr/bin/env python3
"""
apply_diff.py

A tolerant unified-diff applier (no git / no patch utility required).

Features:
- Parses unified diffs (git diff style is OK)
- Strips markdown fences and extra assistant text around the diff
- Applies hunks using context-based matching (not strict line numbers)
- Whitespace-tolerant fallback matching
- Backup original files (*.bak) before writing
- Dry-run mode

Usage:
    python3 apply_diff.py --patch change.diff --repo .
    python3 apply_diff.py --patch change.diff --repo . --dry-run
    python3 apply_diff.py --patch change.diff --repo . --verbose

Tip:
If your patch file contains markdown/code fences from chat output, this script will
usually still work.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


# ----------------------------- Data models -----------------------------

@dataclass
class HunkLine:
    tag: str   # ' ', '-', '+'
    text: str  # includes trailing '\n' if present in patch content


@dataclass
class Hunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    section: str = ""
    lines: List[HunkLine] = field(default_factory=list)


@dataclass
class FilePatch:
    old_path: str
    new_path: str
    hunks: List[Hunk] = field(default_factory=list)


# ----------------------------- Utilities -----------------------------

HUNK_RE = re.compile(
    r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$"
)

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def norm_ws(s: str) -> str:
    # whitespace-tolerant normalization, preserves code tokens better than full collapse
    return " ".join(s.rstrip("\n").rstrip("\r").strip().split())

def strip_prefix_path(p: str) -> str:
    # handles a/foo -> foo and b/foo -> foo
    if p.startswith("a/") or p.startswith("b/"):
        return p[2:]
    return p

def safe_read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="surrogateescape")
    except FileNotFoundError:
        return ""
    except Exception as ex:
        raise RuntimeError(f"Cannot read {path}: {ex}") from ex

def safe_write_text(path: Path, text: str):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", errors="surrogateescape")
    except Exception as ex:
        raise RuntimeError(f"Cannot write {path}: {ex}") from ex


# ----------------------------- Patch sanitizing -----------------------------

def extract_probable_diff(raw: str) -> str:
    """
    Extract the actual diff text from a chat/markdown response.

    Handles cases like:
      ```diff id="..."
      diff --git ...
      ...
      ```
      extra text...
    """
    lines = raw.splitlines(keepends=True)

    # 1) Prefer a ```diff fenced block if present
    in_diff_fence = False
    diff_lines: List[str] = []
    saw_diff_header_inside_fence = False

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("```"):
            if not in_diff_fence:
                # opening fence
                # accept any fence that mentions diff, or generic fence
                if stripped.startswith("```diff") or stripped == "```":
                    in_diff_fence = True
                    continue
            else:
                # closing fence
                if saw_diff_header_inside_fence and diff_lines:
                    return "".join(diff_lines)
                in_diff_fence = False
                diff_lines.clear()
                saw_diff_header_inside_fence = False
                continue

        if in_diff_fence:
            if line.startswith("diff --git ") or line.startswith("--- "):
                saw_diff_header_inside_fence = True
            if saw_diff_header_inside_fence:
                diff_lines.append(line)

    # 2) Fallback: start from first diff header in full text and stop at next fence
    start_idx = None
    for i, line in enumerate(lines):
        if line.startswith("diff --git ") or line.startswith("--- "):
            start_idx = i
            break

    if start_idx is None:
        return raw  # let parser fail with clearer message

    out: List[str] = []
    for line in lines[start_idx:]:
        if line.strip().startswith("```"):
            break
        out.append(line)

    return "".join(out)


# ----------------------------- Unified diff parser -----------------------------

def parse_unified_diff(text: str) -> List[FilePatch]:
    """
    Tolerant parser for unified diff.
    Supports git-style headers and plain ---/+++ headers.
    """
    lines = text.splitlines(keepends=True)
    patches: List[FilePatch] = []
    i = 0
    cur: Optional[FilePatch] = None

    def finalize_cur():
        nonlocal cur
        if cur is not None:
            patches.append(cur)
            cur = None

    while i < len(lines):
        line = lines[i]

        # Git diff header
        if line.startswith("diff --git "):
            finalize_cur()
            # Parse paths from "diff --git a/x b/x"
            m = re.match(r"^diff --git\s+(.+?)\s+(.+?)\s*$", line)
            oldp = ""
            newp = ""
            if m:
                oldp = m.group(1).strip()
                newp = m.group(2).strip()
            cur = FilePatch(old_path=oldp, new_path=newp, hunks=[])
            i += 1
            continue

        # file headers
        if line.startswith("--- "):
            oldp = line[4:].rstrip("\n")
            i += 1
            if i >= len(lines) or not lines[i].startswith("+++ "):
                # malformed-ish; skip
                continue
            newp = lines[i][4:].rstrip("\n")

            # Start a new patch if needed, or update current
            if cur is None:
                cur = FilePatch(old_path=oldp, new_path=newp, hunks=[])
            else:
                cur.old_path = oldp
                cur.new_path = newp
            i += 1
            continue

        # hunk header
        if line.startswith("@@ "):
            if cur is None:
                raise ValueError(f"Hunk found before file header at line {i+1}")

            m = HUNK_RE.match(line.rstrip("\n"))
            if not m:
                raise ValueError(f"Malformed hunk header at line {i+1}: {line.rstrip()}")

            old_start = int(m.group(1))
            old_count = int(m.group(2) or "1")
            new_start = int(m.group(3))
            new_count = int(m.group(4) or "1")
            section = (m.group(5) or "").strip()

            hunk = Hunk(
                old_start=old_start,
                old_count=old_count,
                new_start=new_start,
                new_count=new_count,
                section=section,
                lines=[],
            )
            i += 1

            while i < len(lines):
                l = lines[i]
                if l.startswith(("diff --git ", "--- ", "@@ ")):
                    break
                if l.startswith("\\ No newline at end of file"):
                    # informational marker; ignore
                    i += 1
                    continue
                if not l:
                    i += 1
                    continue

                tag = l[:1]
                if tag in (" ", "-", "+"):
                    hunk.lines.append(HunkLine(tag=tag, text=l[1:]))
                    i += 1
                    continue

                # tolerate junk lines after patch block by stopping hunk if it looks like prose
                if l.strip() and not l.startswith(("index ", "new file mode", "deleted file mode", "similarity index", "rename ")):
                    # If a non-patch line appears, stop this hunk and let outer loop process it
                    break

                i += 1

            cur.hunks.append(hunk)
            continue

        i += 1

    finalize_cur()
    return [p for p in patches if p.hunks]


# ----------------------------- Matching / applying -----------------------------

def hunk_before_after(hunk: Hunk) -> Tuple[List[str], List[str]]:
    before = []
    after = []
    for hl in hunk.lines:
        if hl.tag in (" ", "-"):
            before.append(hl.text)
        if hl.tag in (" ", "+"):
            after.append(hl.text)
    return before, after

def exact_match_at(buf: List[str], pos: int, seq: List[str]) -> bool:
    if pos < 0 or pos + len(seq) > len(buf):
        return False
    return buf[pos:pos+len(seq)] == seq

def norm_match_at(buf: List[str], pos: int, seq: List[str]) -> bool:
    if pos < 0 or pos + len(seq) > len(buf):
        return False
    for a, b in zip(buf[pos:pos+len(seq)], seq):
        if norm_ws(a) != norm_ws(b):
            return False
    return True

def find_best_position(
    buf: List[str],
    seq: List[str],
    preferred_pos: int,
    search_radius: int = 300,
    whitespace_fallback: bool = True,
) -> Tuple[Optional[int], str]:
    """
    Find seq in buf near preferred_pos, then globally, then whitespace-normalized.
    Returns (pos, mode)
    """
    if not seq:
        # pure insertion hunk with no context/deletions: insert at preferred_pos
        pos = max(0, min(preferred_pos, len(buf)))
        return pos, "empty-anchor"

    # 1) exact near preferred
    start = max(0, preferred_pos - search_radius)
    end = min(len(buf) - len(seq), preferred_pos + search_radius)
    for pos in range(start, end + 1):
        if exact_match_at(buf, pos, seq):
            return pos, "exact-near"

    # 2) exact global
    for pos in range(0, len(buf) - len(seq) + 1):
        if exact_match_at(buf, pos, seq):
            return pos, "exact-global"

    if whitespace_fallback:
        # 3) ws-normalized near preferred
        for pos in range(start, end + 1):
            if norm_match_at(buf, pos, seq):
                return pos, "ws-near"

        # 4) ws-normalized global
        for pos in range(0, len(buf) - len(seq) + 1):
            if norm_match_at(buf, pos, seq):
                return pos, "ws-global"

    return None, "not-found"

def apply_hunk_to_buffer(
    buf: List[str],
    hunk: Hunk,
    cumulative_offset: int,
    verbose: bool = False
) -> Tuple[List[str], int]:
    before, after = hunk_before_after(hunk)

    # unified diff line numbers are 1-based; convert to 0-based approx anchor
    preferred_pos = max(0, (hunk.old_start - 1) + cumulative_offset)

    pos, mode = find_best_position(buf, before, preferred_pos)
    if pos is None:
        # Better diagnostics
        sample = "".join(before[:6])
        raise RuntimeError(
            f"Cannot place hunk @@ -{hunk.old_start},{hunk.old_count} +{hunk.new_start},{hunk.new_count} @@ "
            f"(anchor not found). First before-lines sample:\n{sample}"
        )

    if verbose:
        print(f"    hunk @@ -{hunk.old_start},{hunk.old_count} +{hunk.new_start},{hunk.new_count} @@ -> pos {pos} ({mode})")

    new_buf = buf[:pos] + after + buf[pos + len(before):]
    delta = len(after) - len(before)
    return new_buf, cumulative_offset + delta

def resolve_target_paths(fp: FilePatch) -> Tuple[Optional[str], Optional[str]]:
    oldp = fp.old_path.strip()
    newp = fp.new_path.strip()

    # Remove timestamps in plain diff headers if present: "--- path\tdate"
    if "\t" in oldp:
        oldp = oldp.split("\t", 1)[0]
    if "\t" in newp:
        newp = newp.split("\t", 1)[0]

    oldp = strip_prefix_path(oldp)
    newp = strip_prefix_path(newp)

    if oldp == "/dev/null":
        oldp = None
    if newp == "/dev/null":
        newp = None

    return oldp, newp

def apply_file_patch(
    repo: Path,
    fp: FilePatch,
    dry_run: bool = False,
    backup: bool = True,
    verbose: bool = False
) -> None:
    oldp, newp = resolve_target_paths(fp)

    # Determine target file (modifications/creations use new path; deletions use old path)
    target_rel = newp if newp is not None else oldp
    if target_rel is None:
        raise RuntimeError("Patch has neither old nor new path")

    target = (repo / target_rel).resolve()

    # Prevent escaping repo accidentally
    try:
        target.relative_to(repo.resolve())
    except Exception:
        raise RuntimeError(f"Refusing to write outside repo: {target}")

    is_delete = newp is None
    is_create = oldp is None

    original_text = "" if is_create else safe_read_text(target)
    buf = original_text.splitlines(keepends=True)

    if verbose:
        print(f"  file: {target_rel} ({'create' if is_create else 'delete' if is_delete else 'modify'})")

    cumulative_offset = 0
    for h in fp.hunks:
        buf, cumulative_offset = apply_hunk_to_buffer(buf, h, cumulative_offset, verbose=verbose)

    new_text = "".join(buf)

    if is_delete:
        if dry_run:
            print(f"  [DRY] would delete {target_rel}")
            return
        if target.exists():
            if backup:
                bak = target.with_suffix(target.suffix + ".bak")
                safe_write_text(bak, original_text)
            target.unlink()
            print(f"  [OK] deleted {target_rel}")
        else:
            print(f"  [OK] already absent: {target_rel}")
        return

    if dry_run:
        if new_text != original_text:
            print(f"  [DRY] would patch {target_rel}")
        else:
            print(f"  [DRY] no changes for {target_rel}")
        return

    if backup and target.exists():
        bak = target.with_suffix(target.suffix + ".bak")
        safe_write_text(bak, original_text)

    safe_write_text(target, new_text)
    print(f"  [OK] patched {target_rel}")


# ----------------------------- CLI -----------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Tolerant unified diff applier (no git needed)")
    ap.add_argument("--patch", required=True, help="Path to .diff/.patch file")
    ap.add_argument("--repo", default=".", help="Project root (default: current dir)")
    ap.add_argument("--dry-run", action="store_true", help="Parse and simulate only")
    ap.add_argument("--no-backup", action="store_true", help="Do not create .bak files")
    ap.add_argument("--verbose", action="store_true", help="Verbose hunk placement logs")
    args = ap.parse_args()

    patch_path = Path(args.patch).resolve()
    repo = Path(args.repo).resolve()

    if not patch_path.is_file():
        eprint(f"Patch file not found: {patch_path}")
        return 2
    if not repo.is_dir():
        eprint(f"Repo is not a directory: {repo}")
        return 2

    raw = safe_read_text(patch_path)
    cleaned = extract_probable_diff(raw)

    print(f"Patch: {patch_path}")
    print(f"Repo : {repo}")

    try:
        file_patches = parse_unified_diff(cleaned)
    except Exception as ex:
        eprint(f"[ERROR] Failed to parse diff: {ex}")
        # helpful hint
        eprint("Hint: make sure the patch file contains a raw unified diff. "
               "If it was copied from chat, remove markdown fences and extra text.")
        return 1

    if not file_patches:
        eprint("[ERROR] No valid file hunks found in patch.")
        return 1

    print(f"Found {len(file_patches)} file patch(es)")

    failed = 0
    for fp in file_patches:
        try:
            apply_file_patch(
                repo=repo,
                fp=fp,
                dry_run=args.dry_run,
                backup=not args.no_backup,
                verbose=args.verbose,
            )
        except Exception as ex:
            failed += 1
            eprint(f"  [FAIL] {ex}")

    if failed:
        eprint(f"Done with {failed} failure(s).")
        return 1

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
