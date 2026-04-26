"""Register the local retro_integration/ folder with stable-retro."""

from pathlib import Path

import stable_retro as retro

GAME_NAME = "PokemonFireRed-GbAdvance"
INTEGRATION_DIR = Path(__file__).resolve().parent.parent / "retro_integration"


def register_custom_integration() -> None:
    """Make stable-retro look in our repo for the FireRed integration files."""
    retro.data.Integrations.add_custom_path(str(INTEGRATION_DIR))


def integration_path() -> Path:
    return INTEGRATION_DIR / GAME_NAME


def has_start_state() -> bool:
    return (integration_path() / "start.state").exists()


def has_rom() -> bool:
    return any(integration_path().glob("rom.*")) and not all(
        p.suffix == ".sha" for p in integration_path().glob("rom.*")
    )
