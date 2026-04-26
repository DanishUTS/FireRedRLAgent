"""Copies a Pokemon FireRed ROM into the local retro_integration/ folder
so stable-retro can find it. ROMs aren't checked into the repo for legal
reasons — every collaborator runs this once with their own ROM file.

Usage:
    python scripts/import_rom.py /path/to/PokemonFireRed.gba
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

# Allow `python scripts/import_rom.py` to import from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.integrations import integration_path  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("rom", type=Path, help="Path to PokemonFireRed.gba")
    args = p.parse_args()

    if not args.rom.exists():
        print(f"ROM not found: {args.rom}", file=sys.stderr)
        return 1

    sha_file = integration_path() / "rom.sha"
    expected = sha_file.read_text().strip().lower()
    actual = hashlib.sha1(args.rom.read_bytes()).hexdigest().lower()

    if actual != expected:
        print(
            f"ROM SHA1 mismatch.\n  expected: {expected}\n  actual:   {actual}\n"
            "Make sure you're using the unmodified US 1.0 (BPRE) FireRed ROM.",
            file=sys.stderr,
        )
        return 2

    dest = integration_path() / "rom.gba"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.rom, dest)
    print(f"Imported ROM → {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
