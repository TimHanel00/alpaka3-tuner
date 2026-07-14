from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

project = "alpakaTune"
author = "Tim Hanel"
release = "3.0.0"
root_doc = "index"

extensions = ["breathe"]
breathe_projects = {"alpakaTune": str(ROOT / "docs" / "_doxygen" / "xml")}
breathe_default_project = "alpakaTune"

exclude_patterns = ["_build"]
html_theme = "sphinx_rtd_theme"
html_title = "alpakaTune"
