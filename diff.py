#!/usr/bin/env python3
"""
diff.py

A tolerant text-patch applier that does not require git or the system `patch`
utility.

What it supports well:
- Standard unified diffs (`@@ -old,+new @@`)
- Git-style diffs (`diff --git`, `rename from/to`, `new file mode`, etc.)
- Chat/LLM patches that use bare `@@` block markers with no line numbers
- Paths with a/ and b/ prefixes
- New files, deleted files, renames, and copies (text patches)
- Markdown fences and extra prose around the actual patch text
- Context-based matching instead of strict line-number-only application
- Whitespace-tolerant fallback matching
- Optional backup files (`.bak`)
- Dry-run mode

What it does NOT support:
- Git binary patches / literal binary hunks
- Combined merge diffs (`@@@`)
- Classic context diff / normal diff / ed scripts

Usage:
    python3 diff.py --patch change.diff --repo .
    python3 diff.py --patch change.diff --repo . --dry-run --verbose
    python3 diff.py --patch change.diff --repo . --strip 1
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


# ----------------------------- Data models -----------------------------

@dataclass
class HunkLine:
    tag: str   # ' ', '-', '+'
    text: str  # includes trailing '\n' if present in patch content


@dataclass
class Hunk:
    old_start: Optional[int] = None
    old_count: Optional[int] = None
    new_start: Optional[int] = None
    new_count: Optional[int] = None
    section: str = ""
    lines: List[HunkLine] = field(default_factory=list)
    header_style: str = "unified"   # unified | bare


@dataclass
class FilePatch:
    old_path: Optional[str] = None
    new_path: Optional[str] = None
    hunks: List[Hunk] = field(default_factory=list)
    is_new: bool = False
    is_delete: bool = False
    is_rename: bool = False
    is_copy: bool = False
    rename_from: Optional[str] = None
    rename_to: Optional[str] = None
    copy_from: Optional[str] = None
    copy_to: Optional[str] = None
    is_binary: bool = False
    metadata: List[str] = field(default_factory=list)


# ----------------------------- Utilities -----------------------------

STANDARD_HUNK_RE = re.compile(
    r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$"
)

LIKELY_PATCH_PREFIXES = (
    "diff --git ",
    "--- ",
    "Index: ",
    "*** ",
    "Binary files ",
    "GIT binary patch",
    "rename from ",
    "rename to ",
    "new file mode ",
    "deleted file mode ",
)

PATCH_META_PREFIXES = (
    "index ",
    "old mode ",
    "new mode ",
    "deleted file mode ",
    "new file mode ",
    "similarity index ",
    "dissimilarity index ",
    "rename from ",
    "rename to ",
    "copy from ",
    "copy to ",
    "Binary files ",
    "GIT binary patch",
)



def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)



def norm_ws(s: str) -> str:
    return " ".join(s.rstrip("\n").rstrip("\r").strip().split())



def strip_known_prefixes(path_text: str, strip_components: int = 0) -> str:
    p = path_text.strip()
    if p.startswith("a/") or p.startswith("b/"):
        p = p[2:]
    if strip_components > 0:
        parts = p.split("/")
        if len(parts) > strip_components:
            p = "/".join(parts[strip_components:])
        else:
            p = parts[-1] if parts else p
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



def backup_file(path: Path):
    if not path.exists():
        return
    bak = path.with_suffix(path.suffix + ".bak")
    safe_write_text(bak, safe_read_text(path))



def safe_copy_file(src: Path, dst: Path):
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    except Exception as ex:
        raise RuntimeError(f"Cannot copy {src} -> {dst}: {ex}") from ex



def remove_timestamp_suffix(p: Optional[str]) -> Optional[str]:
    if p is None:
        return None
    if "\t" in p:
        p = p.split("\t", 1)[0]
    return p.strip()



def infer_hunk_counts(hunk: Hunk) -> None:
    if hunk.old_count is None:
        hunk.old_count = sum(1 for hl in hunk.lines if hl.tag in (" ", "-"))
    if hunk.new_count is None:
        hunk.new_count = sum(1 for hl in hunk.lines if hl.tag in (" ", "+"))
    if hunk.old_start is None:
        hunk.old_start = 1
    if hunk.new_start is None:
        hunk.new_start = 1



def path_within_repo(path: Path, repo: Path) -> bool:
    try:
        path.resolve().relative_to(repo.resolve())
        return True
    except Exception:
        return False


# ----------------------------- Patch sanitizing -----------------------------


def is_likely_patch_line(line: str) -> bool:
    if line.startswith("@@"):
        return True
    return line.startswith(LIKELY_PATCH_PREFIXES)



def extract_probable_diff(raw: str) -> str:
    """
    Extract the actual diff text from a chat/markdown response.
    Keeps text from the first likely patch header onward, and prefers fenced
    code blocks that actually contain patch markers.
    """
    lines = raw.splitlines(keepends=True)

    # Prefer fenced blocks that contain likely patch lines.
    in_fence = False
    fence_buf: List[str] = []
    found_patch_inside = False
    best_block: Optional[str] = None

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            if not in_fence:
                in_fence = True
                fence_buf = []
                found_patch_inside = False
            else:
                if found_patch_inside and fence_buf:
                    best_block = "".join(fence_buf)
                    break
                in_fence = False
                fence_buf = []
                found_patch_inside = False
            continue

        if in_fence:
            if is_likely_patch_line(line):
                found_patch_inside = True
            if found_patch_inside:
                fence_buf.append(line)

    if best_block:
        return best_block

    start_idx = None
    for i, line in enumerate(lines):
        if is_likely_patch_line(line):
            start_idx = i
            break

    if start_idx is None:
        return raw

    out: List[str] = []
    for line in lines[start_idx:]:
        if line.strip().startswith("```"):
            continue
        out.append(line)
    return "".join(out)


# ----------------------------- Hunk helpers -----------------------------


def parse_hunk_header(line: str) -> Optional[Hunk]:
    s = line.rstrip("\n")

    if s.startswith("@@@"):
        raise ValueError("Combined merge diffs (@@@) are not supported")

    m = STANDARD_HUNK_RE.match(s)
    if m:
        return Hunk(
            old_start=int(m.group(1)),
            old_count=int(m.group(2) or "1"),
            new_start=int(m.group(3)),
            new_count=int(m.group(4) or "1"),
            section=(m.group(5) or "").strip(),
            lines=[],
            header_style="unified",
        )

    if s.strip() == "@@":
        return Hunk(
            old_start=None,
            old_count=None,
            new_start=None,
            new_count=None,
            section="",
            lines=[],
            header_style="bare",
        )

    return None



def hunk_before_after(hunk: Hunk) -> Tuple[List[str], List[str]]:
    before: List[str] = []
    after: List[str] = []
    for hl in hunk.lines:
        if hl.tag in (" ", "-"):
            before.append(hl.text)
        if hl.tag in (" ", "+"):
            after.append(hl.text)
    return before, after



def hunk_context_prefix_suffix(hunk: Hunk) -> Tuple[List[str], List[str]]:
    prefix: List[str] = []
    suffix: List[str] = []
    i = 0
    while i < len(hunk.lines) and hunk.lines[i].tag == " ":
        prefix.append(hunk.lines[i].text)
        i += 1
    j = len(hunk.lines) - 1
    while j >= 0 and hunk.lines[j].tag == " ":
        suffix.append(hunk.lines[j].text)
        j -= 1
    suffix.reverse()
    return prefix, suffix


# ----------------------------- Unified diff parser -----------------------------


def parse_unified_diff(text: str) -> List[FilePatch]:
    """
    Tolerant parser for real-world text patches.
    Supports:
    - git diff headers
    - plain --- / +++ headers
    - bare @@ hunk blocks with no explicit ranges
    - rename/copy/new/delete metadata
    """
    lines = text.splitlines(keepends=True)
    patches: List[FilePatch] = []
    i = 0
    cur: Optional[FilePatch] = None

    def finalize_cur() -> None:
        nonlocal cur
        if cur is not None:
            if cur.hunks or cur.is_new or cur.is_delete or cur.is_rename or cur.is_copy or cur.is_binary:
                patches.append(cur)
            cur = None

    while i < len(lines):
        line = lines[i]

        if line.startswith("diff --git "):
            finalize_cur()
            cur = FilePatch(metadata=[line.rstrip("\n")])
            i += 1
            continue

        if line.startswith("Index: "):
            finalize_cur()
            idx_path = line[len("Index: "):].rstrip("\n")
            cur = FilePatch(old_path=idx_path, new_path=idx_path, metadata=[line.rstrip("\n")])
            i += 1
            continue

        if any(line.startswith(p) for p in PATCH_META_PREFIXES):
            if cur is None:
                cur = FilePatch()
            cur.metadata.append(line.rstrip("\n"))

            s = line.rstrip("\n")
            if s.startswith("new file mode "):
                cur.is_new = True
            elif s.startswith("deleted file mode "):
                cur.is_delete = True
            elif s.startswith("rename from "):
                cur.is_rename = True
                cur.rename_from = s[len("rename from "):].strip()
                if cur.old_path is None:
                    cur.old_path = cur.rename_from
            elif s.startswith("rename to "):
                cur.is_rename = True
                cur.rename_to = s[len("rename to "):].strip()
                if cur.new_path is None:
                    cur.new_path = cur.rename_to
            elif s.startswith("copy from "):
                cur.is_copy = True
                cur.copy_from = s[len("copy from "):].strip()
                if cur.old_path is None:
                    cur.old_path = cur.copy_from
            elif s.startswith("copy to "):
                cur.is_copy = True
                cur.copy_to = s[len("copy to "):].strip()
                if cur.new_path is None:
                    cur.new_path = cur.copy_to
            elif s.startswith("Binary files ") or s == "GIT binary patch":
                cur.is_binary = True
            i += 1
            continue

        if line.startswith("--- "):
            oldp = remove_timestamp_suffix(line[4:].rstrip("\n"))
            i += 1
            if i >= len(lines) or not lines[i].startswith("+++ "):
                continue
            newp = remove_timestamp_suffix(lines[i][4:].rstrip("\n"))

            if cur is None:
                cur = FilePatch(old_path=oldp, new_path=newp)
            else:
                cur.old_path = oldp
                cur.new_path = newp

            if oldp == "/dev/null":
                cur.is_new = True
            if newp == "/dev/null":
                cur.is_delete = True
            i += 1
            continue

        if line.startswith("@@"):
            if cur is None:
                raise ValueError(f"Hunk found before file header at line {i+1}")

            hunk = parse_hunk_header(line)
            if hunk is None:
                raise ValueError(f"Malformed hunk header at line {i+1}: {line.rstrip()}")
            i += 1

            while i < len(lines):
                l = lines[i]
                if l.startswith(("diff --git ", "Index: ", "--- ", "@@")):
                    break
                if l.startswith("\\ No newline at end of file"):
                    i += 1
                    continue
                tag = l[:1]
                if tag in (" ", "-", "+"):
                    hunk.lines.append(HunkLine(tag=tag, text=l[1:]))
                    i += 1
                    continue
                # Skip patch metadata that might appear between blocks.
                if any(l.startswith(p) for p in PATCH_META_PREFIXES):
                    break
                # Non-patch prose: stop the hunk and let outer loop continue.
                if l.strip():
                    break
                i += 1

            infer_hunk_counts(hunk)
            cur.hunks.append(hunk)
            continue

        i += 1

    finalize_cur()
    return [p for p in patches if p.hunks or p.is_new or p.is_delete or p.is_rename or p.is_copy or p.is_binary]


# ----------------------------- Matching / applying -----------------------------


def exact_match_at(buf: Sequence[str], pos: int, seq: Sequence[str]) -> bool:
    if pos < 0 or pos + len(seq) > len(buf):
        return False
    return list(buf[pos:pos + len(seq)]) == list(seq)



def norm_match_at(buf: Sequence[str], pos: int, seq: Sequence[str]) -> bool:
    if pos < 0 or pos + len(seq) > len(buf):
        return False
    for a, b in zip(buf[pos:pos + len(seq)], seq):
        if norm_ws(a) != norm_ws(b):
            return False
    return True



def find_all_positions(buf: Sequence[str], seq: Sequence[str], normalized: bool) -> List[int]:
    if not seq:
        return list(range(0, len(buf) + 1))
    out: List[int] = []
    for pos in range(0, len(buf) - len(seq) + 1):
        ok = norm_match_at(buf, pos, seq) if normalized else exact_match_at(buf, pos, seq)
        if ok:
            out.append(pos)
    return out



def distance_score(pos: int, preferred_pos: int) -> int:
    return abs(pos - preferred_pos)



def choose_best_position(candidates: Iterable[int], preferred_pos: int) -> Optional[int]:
    best: Optional[int] = None
    best_score: Optional[int] = None
    for pos in candidates:
        sc = distance_score(pos, preferred_pos)
        if best is None or sc < best_score:  # type: ignore[operator]
            best = pos
            best_score = sc
    return best



def find_position_by_context(
    buf: Sequence[str],
    hunk: Hunk,
    preferred_pos: int,
    normalized: bool,
) -> Tuple[Optional[int], str]:
    prefix, suffix = hunk_context_prefix_suffix(hunk)
    before, _ = hunk_before_after(hunk)

    if not prefix and not suffix:
        return None, "no-context"

    def matcher(seq_a: Sequence[str], seq_b: Sequence[str]) -> bool:
        if len(seq_a) != len(seq_b):
            return False
        for a, b in zip(seq_a, seq_b):
            if normalized:
                if norm_ws(a) != norm_ws(b):
                    return False
            else:
                if a != b:
                    return False
        return True

    candidates: List[int] = []
    buf_len = len(buf)

    if prefix and suffix:
        # Look for prefix and suffix in order, then infer start.
        prefix_positions = find_all_positions(buf, prefix, normalized)
        suffix_positions = find_all_positions(buf, suffix, normalized)
        for p in prefix_positions:
            prefix_end = p + len(prefix)
            for s in suffix_positions:
                if s < prefix_end:
                    continue
                inferred_start = p
                # If before exists, try to approximate the full replacement width.
                if inferred_start < 0 or inferred_start > buf_len:
                    continue
                candidates.append(inferred_start)
    elif prefix:
        candidates.extend(find_all_positions(buf, prefix, normalized))
    elif suffix:
        for s in find_all_positions(buf, suffix, normalized):
            candidates.append(s)

    if not candidates:
        return None, "context-not-found"

    best = choose_best_position(candidates, preferred_pos)
    if best is None:
        return None, "context-not-found"

    # For suffix-only anchor, start should be before suffix based on full-before width.
    if suffix and not prefix and before:
        best = max(0, best - (len(before) - len(suffix)))
    return best, "context-ws" if normalized else "context-exact"



def find_best_position(
    buf: Sequence[str],
    hunk: Hunk,
    preferred_pos: int,
    search_radius: int = 300,
    whitespace_fallback: bool = True,
) -> Tuple[Optional[int], str]:
    before, _ = hunk_before_after(hunk)

    if not before:
        pos = max(0, min(preferred_pos, len(buf)))
        return pos, "empty-anchor"

    # 1) Exact near preferred.
    start = max(0, preferred_pos - search_radius)
    end = min(len(buf) - len(before), preferred_pos + search_radius)
    for pos in range(start, end + 1):
        if exact_match_at(buf, pos, before):
            return pos, "exact-near"

    # 2) Exact global.
    for pos in range(0, len(buf) - len(before) + 1):
        if exact_match_at(buf, pos, before):
            return pos, "exact-global"

    # 3) Context-only exact fallback.
    pos, mode = find_position_by_context(buf, hunk, preferred_pos, normalized=False)
    if pos is not None:
        return pos, mode

    if whitespace_fallback:
        # 4) WS-normalized near preferred.
        for pos in range(start, end + 1):
            if norm_match_at(buf, pos, before):
                return pos, "ws-near"

        # 5) WS-normalized global.
        for pos in range(0, len(buf) - len(before) + 1):
            if norm_match_at(buf, pos, before):
                return pos, "ws-global"

        # 6) Context-only WS fallback.
        pos, mode = find_position_by_context(buf, hunk, preferred_pos, normalized=True)
        if pos is not None:
            return pos, mode

    return None, "not-found"



def apply_hunk_to_buffer(
    buf: List[str],
    hunk: Hunk,
    cumulative_offset: int,
    verbose: bool = False,
) -> Tuple[List[str], int]:
    infer_hunk_counts(hunk)
    before, after = hunk_before_after(hunk)

    preferred_old_start = hunk.old_start or 1
    preferred_pos = max(0, (preferred_old_start - 1) + cumulative_offset)

    pos, mode = find_best_position(buf, hunk, preferred_pos)
    if pos is None:
        sample = "".join(before[:8]) if before else "<insertion hunk>"
        raise RuntimeError(
            f"Cannot place hunk ({hunk.header_style}) near old_start={hunk.old_start or 1}. "
            f"Anchor not found. First before-lines sample:\n{sample}"
        )

    if verbose:
        print(
            f"    hunk old={hunk.old_start},{hunk.old_count} new={hunk.new_start},{hunk.new_count} "
            f"-> pos {pos} ({mode})"
        )

    new_buf = buf[:pos] + after + buf[pos + len(before):]
    delta = len(after) - len(before)
    return new_buf, cumulative_offset + delta


# ----------------------------- Path resolution -----------------------------


def resolve_target_paths(fp: FilePatch, strip_components: int) -> Tuple[Optional[str], Optional[str]]:
    oldp = remove_timestamp_suffix(fp.old_path)
    newp = remove_timestamp_suffix(fp.new_path)

    if fp.rename_from:
        oldp = fp.rename_from
    if fp.rename_to:
        newp = fp.rename_to
    if fp.copy_from:
        oldp = fp.copy_from
    if fp.copy_to:
        newp = fp.copy_to

    if oldp is not None:
        oldp = strip_known_prefixes(oldp, strip_components)
    if newp is not None:
        newp = strip_known_prefixes(newp, strip_components)

    if oldp == "/dev/null":
        oldp = None
    if newp == "/dev/null":
        newp = None

    return oldp, newp


# ----------------------------- Patch application -----------------------------


def load_base_content(
    repo: Path,
    old_rel: Optional[str],
    new_rel: Optional[str],
    fp: FilePatch,
) -> Tuple[str, Optional[Path], Optional[Path]]:
    old_path = (repo / old_rel).resolve() if old_rel else None
    new_path = (repo / new_rel).resolve() if new_rel else None

    if old_path and not path_within_repo(old_path, repo):
        raise RuntimeError(f"Refusing to read outside repo: {old_path}")
    if new_path and not path_within_repo(new_path, repo):
        raise RuntimeError(f"Refusing to write outside repo: {new_path}")

    base_text = ""

    # Prefer old path for renames, copies, deletes, and modifications.
    if old_path and old_path.exists():
        base_text = safe_read_text(old_path)
    elif new_path and new_path.exists():
        base_text = safe_read_text(new_path)
    elif fp.is_new or old_rel is None:
        base_text = ""
    else:
        base_text = ""

    return base_text, old_path, new_path



def apply_file_patch(
    repo: Path,
    fp: FilePatch,
    dry_run: bool = False,
    backup: bool = True,
    verbose: bool = False,
    strip_components: int = 0,
) -> None:
    if fp.is_binary:
        raise RuntimeError("Binary patch detected; only text patches are supported")

    old_rel, new_rel = resolve_target_paths(fp, strip_components)
    base_text, old_path, new_path = load_base_content(repo, old_rel, new_rel, fp)

    target_rel = new_rel if new_rel is not None else old_rel
    target_path = new_path if new_path is not None else old_path
    if target_rel is None or target_path is None:
        raise RuntimeError("Patch has neither old nor new path")

    is_delete = fp.is_delete or (new_rel is None)
    is_create = fp.is_new or (old_rel is None)
    is_rename = fp.is_rename and old_rel is not None and new_rel is not None and old_rel != new_rel
    is_copy = fp.is_copy and old_rel is not None and new_rel is not None and old_rel != new_rel

    buf = base_text.splitlines(keepends=True)

    if verbose:
        kind = "modify"
        if is_create:
            kind = "create"
        elif is_delete:
            kind = "delete"
        elif is_rename:
            kind = "rename"
        elif is_copy:
            kind = "copy"
        print(f"  file: {target_rel} ({kind})")

    cumulative_offset = 0
    for h in fp.hunks:
        buf, cumulative_offset = apply_hunk_to_buffer(buf, h, cumulative_offset, verbose=verbose)

    new_text = "".join(buf)

    if dry_run:
        if is_delete:
            print(f"  [DRY] would delete {old_rel or target_rel}")
            return
        if is_rename:
            print(f"  [DRY] would rename {old_rel} -> {new_rel}")
            if fp.hunks:
                print(f"  [DRY] would patch {new_rel}")
            return
        if is_copy:
            print(f"  [DRY] would copy {old_rel} -> {new_rel}")
            if fp.hunks:
                print(f"  [DRY] would patch {new_rel}")
            return
        if target_path.exists() and new_text == safe_read_text(target_path):
            print(f"  [DRY] no changes for {target_rel}")
        else:
            print(f"  [DRY] would patch {target_rel}")
        return

    if is_delete:
        if old_path is None:
            raise RuntimeError("Delete patch has no source path")
        if old_path.exists():
            if backup:
                backup_file(old_path)
            old_path.unlink()
            print(f"  [OK] deleted {old_rel}")
        else:
            print(f"  [OK] already absent: {old_rel}")
        return

    if is_rename:
        assert old_path is not None and new_path is not None
        if backup and old_path.exists():
            backup_file(old_path)
        if backup and new_path.exists():
            backup_file(new_path)
        safe_write_text(new_path, new_text)
        if old_path.exists() and old_path != new_path:
            old_path.unlink()
        print(f"  [OK] renamed {old_rel} -> {new_rel}")
        return

    if is_copy:
        assert new_path is not None
        if backup and new_path.exists():
            backup_file(new_path)
        safe_write_text(new_path, new_text)
        print(f"  [OK] copied {old_rel} -> {new_rel}")
        return

    if backup and target_path.exists():
        backup_file(target_path)

    safe_write_text(target_path, new_text)
    if is_create and not fp.hunks and old_path and old_path.exists() and target_path != old_path:
        safe_copy_file(old_path, target_path)
    print(f"  [OK] patched {target_rel}")


# ----------------------------- CLI -----------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="Tolerant text diff applier (no git needed)")
    ap.add_argument("--patch", required=True, help="Path to .diff/.patch file")
    ap.add_argument("--repo", default=".", help="Project root (default: current dir)")
    ap.add_argument("--dry-run", action="store_true", help="Parse and simulate only")
    ap.add_argument("--no-backup", action="store_true", help="Do not create .bak files")
    ap.add_argument("--verbose", action="store_true", help="Verbose placement logs")
    ap.add_argument(
        "--strip",
        type=int,
        default=0,
        help="Strip N leading path components after removing a/ or b/ prefixes",
    )
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
        eprint(
            "Hint: supply a text patch in unified/git diff form. "
            "Markdown fences and extra chat text are OK."
        )
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
                strip_components=args.strip,
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
