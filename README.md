# tolerant-diff-applier

A tolerant unified-diff applier in Python — designed for messy AI/chat-generated patches that `git apply` often rejects.

It applies unified diffs **without requiring `git` or the `patch` utility**, and it can handle common copy/paste problems like **Markdown code fences**, **extra assistant text**, and **hunks that no longer match exact line numbers**.

> **Built for real-world patch copy/paste chaos.**

---

## Why this exists

`git apply` is excellent when your patch is clean and generated inside a normal Git workflow.

But AI/chat-generated patch output is often messy:

- wrapped in ```diff fences
- mixed with prose / explanation text
- slightly shifted context
- whitespace differences
- copied into `.diff` files with extra noise

This tool is a **forgiving fallback**:
- it extracts probable diff content from chat output
- parses unified diffs (including git-style diff headers)
- applies hunks using **context-based matching**, not just strict line numbers
- falls back to **whitespace-tolerant matching**
- creates **`.bak` backups** before writing
- supports **dry-run mode**

---

## Features

- ✅ Parses **unified diffs** (git diff style supported)
- ✅ Strips **Markdown fences** and extra assistant text around diffs
- ✅ Applies hunks using **context-based matching**
- ✅ **Whitespace-tolerant** fallback matching
- ✅ Creates **backup files** (`*.bak`) before modifying files
- ✅ **Dry-run** mode to preview changes
- ✅ Pure Python (no `git`, no `patch` binary required)
- ✅ Prevents writing **outside the target repo root**

---

## When to use it

Use this when:

- `git apply` fails on a patch copied from ChatGPT/Claude/etc.
- you only have Python installed (no `patch` utility)
- you want a **safer patch preview** with `--dry-run`
- you need a quick way to apply small/medium diffs in scripts or tooling

---

## Install

No package install required — it’s a single script.

```bash
python3 apply_diff.py --help
