from __future__ import annotations

from pathlib import Path

# --- Paths
HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]                 # <repo>/
DOCS_SOURCE = HERE.parent              # <repo>/docs/source
DOXY_XML = ROOT / "docs" / "_doxygen" / "xml"

# --- Try to reuse Alpaka’s Sphinx configuration (theme/extensions/etc.)
ALPAKA_CONF = ROOT / "alpaka" / "docs" / "source" / "conf.py"
if ALPAKA_CONF.exists():
    exec(ALPAKA_CONF.read_text(encoding="utf-8"), globals())

# --- Override / set your project identity
project = "My Alpaka Extension"
html_title = project
author = "You"
root_doc = "index"

# --- Make sure Breathe points at YOUR generated Doxygen XML
extensions = list(dict.fromkeys(list(globals().get("extensions", [])) + ["breathe"]))

breathe_projects = {project: str(DOXY_XML)}
breathe_default_project = project

# --- Keep Sphinx sane
exclude_patterns = list(dict.fromkeys(list(globals().get("exclude_patterns", [])) + [
    "_build",
    "**/.DS_Store",
]))

# Optional: if Alpaka conf didn’t set a theme (fallback)
html_theme = globals().get("html_theme", "sphinx_rtd_theme")
