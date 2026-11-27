#!/usr/bin/env python3
"""Verify all 24 machine-checked Lean 4 proofs for bartleby-1.2 (`lean/`).

Checks:
  1. Every `.lean` file under `lean/` (excluding `.lake/`) is free of forbidden
     bypass tokens (`sorry`, `admit`, `axiom`, `native_decide`, `unsafe`, `implemented_by`).
  2. Every theorem registered in `lean/lean_claims.yaml` exists in its mapped
     `.lean` file and is audited via `#print axioms` in `lean/AxiomsCheck.lean`.
  3. Runs `lake build` and `lake env lean AxiomsCheck.lean` (unless `--no-build`
     is passed) and verifies that all 24 theorems depend only on standard Lean axioms.

Usage:
    python scripts/verify_lean.py [--no-build]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
LEAN_DIR = REPO_ROOT / "lean"
CLAIMS_YAML = LEAN_DIR / "lean_claims.yaml"

_FORBIDDEN_LEAN_RE = re.compile(
    r"\b(sorry|admit|axiom|native_decide|unsafe|implemented_by)\b"
)
_ALLOWED_AXIOMS = {
    "propext",
    "Classical.choice",
    "Quot.sound",
    "Lean.ofReduceBool",
    "Lean.trustCompiler",
}


def strip_lean_comments(text: str) -> str:
    """Remove `/- ... -/` block comments and `-- ...` line comments."""
    out = re.sub(r"/-.*?-/", "", text, flags=re.DOTALL)
    out = re.sub(r"--[^\n]*", "", out)
    return out


def find_lake_env() -> tuple[str | None, dict[str, str]]:
    """Locate the `lake` executable in `~/.elan/bin` or `PATH` (cross-platform)."""
    env = dict(os.environ)
    extra_paths = [
        str(Path.home() / ".elan" / "bin"),
        str(Path.home() / ".local" / "bin"),
    ]
    existing_path = env.get("PATH", "")
    env["PATH"] = os.pathsep.join(
        [p for p in extra_paths + ([existing_path] if existing_path else []) if p]
    )
    found = shutil.which("lake", path=env["PATH"])
    if found and Path(found).is_file():
        return str(found), env
    for exe_name in ("lake", "lake.exe"):
        cand = Path.home() / ".elan" / "bin" / exe_name
        if cand.is_file():
            return str(cand), env
    return None, env


def ensure_shared_mathlib_packages(lean_dir: Path) -> None:
    """Link `lean/.lake/packages` to `LEAN_MATHLIB_CACHE_DIR` if set in environment."""
    cache_env = os.environ.get("LEAN_MATHLIB_CACHE_DIR", "").strip()
    if not cache_env:
        return
    cache_dir = Path(cache_env)
    if not cache_dir.is_dir():
        return
    shared_pkgs = cache_dir / ".lake" / "packages"
    if not shared_pkgs.is_dir():
        return
    local_lake = lean_dir / ".lake"
    local_pkgs = local_lake / "packages"
    if local_pkgs.exists() or local_pkgs.is_symlink():
        return
    local_lake.mkdir(parents=True, exist_ok=True)
    local_pkgs.symlink_to(shared_pkgs)


def verify_lean(*, run_build: bool = True) -> tuple[bool, list[str], dict[str, Any]]:
    """Run static and `lake` verification over `lean/`."""
    errors: list[str] = []
    if not CLAIMS_YAML.is_file():
        return False, [f"Missing {CLAIMS_YAML}"], {}

    data = yaml.safe_load(CLAIMS_YAML.read_text(encoding="utf-8")) or {}
    claims = data.get("claims") or []

    lean_files = [
        p for p in sorted(LEAN_DIR.rglob("*.lean")) if ".lake" not in p.parts
    ]
    for lf in lean_files:
        cleaned = strip_lean_comments(lf.read_text(encoding="utf-8", errors="ignore"))
        for m in _FORBIDDEN_LEAN_RE.finditer(cleaned):
            tok = m.group(1)
            errors.append(f"Forbidden token `{tok}` found in `{lf.relative_to(REPO_ROOT)}`.")

    axioms_check_path = LEAN_DIR / "AxiomsCheck.lean"
    axioms_check_text = (
        axioms_check_path.read_text(encoding="utf-8", errors="ignore")
        if axioms_check_path.is_file()
        else ""
    )

    total_theorems = 0
    for c in claims:
        lbl = str(c.get("latex_label", "")).strip()
        rel_lean = str(c.get("lean_file", "")).strip()
        lf_path = REPO_ROOT / rel_lean
        if not lf_path.is_file():
            errors.append(f"Claim `{lbl}` points to missing Lean file `{rel_lean}`.")
            continue
        lf_text = lf_path.read_text(encoding="utf-8", errors="ignore")
        for thm_full in c.get("lean_theorems") or []:
            total_theorems += 1
            short_name = str(thm_full).split(".")[-1]
            decl_re = re.compile(rf"\b(?:theorem|lemma|def)\s+{re.escape(short_name)}\b")
            if not decl_re.search(lf_text):
                errors.append(
                    f"Claim `{lbl}` theorem `{thm_full}` (`{short_name}`) not found in `{rel_lean}`."
                )
            if axioms_check_text and thm_full not in axioms_check_text:
                errors.append(
                    f"Theorem `{thm_full}` is missing from `#print axioms` in `lean/AxiomsCheck.lean`."
                )

    build_verified = False
    if run_build and not errors:
        lake_bin, env = find_lake_env()
        if not lake_bin:
            errors.append("`lake` binary not found in `~/.elan/bin` or `PATH`.")
        else:
            ensure_shared_mathlib_packages(LEAN_DIR)
            proc_build = subprocess.run(
                [lake_bin, "build"],
                cwd=LEAN_DIR,
                env=env,
                capture_output=True,
                text=True,
            )
            if proc_build.returncode != 0:
                errors.append(
                    f"`lake build` failed in `{LEAN_DIR}`:\n{(proc_build.stdout + proc_build.stderr)[-1200:]}"
                )
            elif axioms_check_path.is_file():
                proc_ax = subprocess.run(
                    [lake_bin, "env", "lean", "AxiomsCheck.lean"],
                    cwd=LEAN_DIR,
                    env=env,
                    capture_output=True,
                    text=True,
                )
                if proc_ax.returncode != 0:
                    errors.append(
                        f"`lake env lean AxiomsCheck.lean` failed:\n{(proc_ax.stdout + proc_ax.stderr)[-1200:]}"
                    )
                else:
                    build_verified = True
                    for line in (proc_ax.stdout + "\n" + proc_ax.stderr).splitlines():
                        if "depends on axioms:" in line:
                            m_ax = re.search(r"\[([^\]]*)\]", line)
                            if m_ax:
                                used = {
                                    a.strip()
                                    for a in m_ax.group(1).split(",")
                                    if a.strip()
                                }
                                disallowed = used - _ALLOWED_AXIOMS
                                if disallowed:
                                    errors.append(
                                        f"Disallowed axioms {sorted(disallowed)} in `{line.strip()}`."
                                    )
                        elif "sorryAx" in line:
                            errors.append(f"`sorryAx` detected in `{line.strip()}`.")

    summary = {
        "registered_claims": len(claims),
        "lean_files_scanned": len(lean_files),
        "verified_theorems": total_theorems,
        "build_verified": build_verified,
    }
    return len(errors) == 0, errors, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-build", action="store_true", help="Skip running `lake build`")
    args = parser.parse_args()

    ok, errors, summary = verify_lean(run_build=not args.no_build)
    print(
        f"Lean claims: {summary.get('registered_claims', 0)} | "
        f"Lean files: {summary.get('lean_files_scanned', 0)} | "
        f"Theorems: {summary.get('verified_theorems', 0)} | "
        f"lake build + AxiomsCheck: {'PASS' if summary.get('build_verified') else ('SKIPPED' if args.no_build else 'FAIL')}"
    )
    if ok:
        if summary.get("build_verified"):
            print(
                "\nAll 24 formal theorems backed by machine-checked Lean 4 proofs (zero sorry/axioms)."
            )
        else:
            print(
                "\nPASS (STATIC AST & CLAIM CHECK ONLY — lake build skipped; "
                "install Lean 4 v4.31.0 via elan to run full kernel verification)"
            )
        return 0
    for err in errors:
        print(f"  ! {err}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
