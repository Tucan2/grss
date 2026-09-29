#!/usr/bin/env python3
"""Build the standalone exhaustive GRSS C++ -> Python binding in-place."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BINDING = ROOT / "grss_full_bindings.cpp"
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
OUT = ROOT / f"grss_full{EXT_SUFFIX}"


def find_cxx() -> str:
    candidates = []
    if os.environ.get("CXX"):
        candidates.append(os.environ["CXX"])
    if sys.platform == "darwin":
        candidates += ["g++-16", "g++-15", "g++-14", "g++-13", "/opt/homebrew/bin/g++-16"]
    else:
        candidates += ["g++", "c++"]
    for c in candidates:
        p = shutil.which(c) if "/" not in c else c
        if p and Path(p).exists():
            out = subprocess.check_output([p, "--version"], text=True, stderr=subprocess.STDOUT)
            if "Apple clang" not in out and "clang" not in out.lower():
                return p
    raise RuntimeError("Could not find a GNU GCC C++ compiler with OpenMP. Set CXX explicitly.")


def main() -> None:
    print(f"Python:  {sys.executable}")
    print(f"Version: {sys.version.split()[0]}")
    print(f"Root:    {ROOT}")
    print(f"Source:  {BINDING}")

    if not BINDING.exists():
        raise FileNotFoundError(f"Missing {BINDING}. Put grss_full_bindings.cpp in the repository root.")

    try:
        import pybind11
    except ImportError as exc:
        raise RuntimeError(
            f"pybind11 is not installed in {sys.executable}. Run:\n"
            f"  {sys.executable} -m pip install pybind11"
        ) from exc

    cxx = find_cxx()
    pybind_inc = Path(pybind11.get_include())
    python_inc = Path(sysconfig.get_paths()["include"])
    include_dir = ROOT / "include"

    cmd = [
        cxx,
        "-std=c++11",
        "-O2",
        "-fPIC",
        "-dynamiclib" if sys.platform == "darwin" else "-shared",
        "-fopenmp",
        f"-I{python_inc}",
        f"-I{pybind_inc}",
        f"-I{include_dir}",
        str(BINDING),
    ]
    if sys.platform == "darwin":
        cmd += ["-Wl,-undefined,dynamic_lookup"]
    cmd += ["-o", str(OUT)]

    print("Building:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)
    print(f"Built: {OUT}")

    sys.path.insert(0, str(ROOT))
    import grss_full
    print("Imported grss_full successfully.")
    print("Example:", grss_full.rad_to_deg(3.141592653589793))


if __name__ == "__main__":
    main()
