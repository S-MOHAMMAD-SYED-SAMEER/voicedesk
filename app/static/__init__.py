"""Static files for the development harness.

One page, read from disk at request time rather than baked into a string, so
it can be edited and reloaded while working on the audio path.
"""

import pathlib

HARNESS_HTML = pathlib.Path(__file__).parent / "harness.html"


def harness_page() -> str:
    return HARNESS_HTML.read_text(encoding="utf-8")
