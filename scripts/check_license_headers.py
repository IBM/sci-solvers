"""Pre-commit check: fail if a header-requiring source file lacks the license header.

Invoked by the ``license-headers`` local pre-commit hook, which passes the staged
filenames as argv. The hook's ``types_or`` / ``files`` / ``exclude`` in
``.pre-commit-config.yaml`` decides *which* files reach this script; here we only
assert the marker is present and report the ones that are missing it, so no
header-requiring file can be committed without a header.

The marker is the SPDX line of ``LICENSE.tmpl`` -- matching just the
marker (not the whole banner) keeps the check robust to the ``${years}`` field and
to Python's ``__copyright__ = \"\"\"..."\"\"`` vs. the ``#``-comment banner form.

Run standalone over everything with::

    uv run python scripts/check_license_headers.py $(git ls-files)
"""

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""

import sys
from pathlib import Path

# The distinctive line every header form shares (Python __copyright__ and the
# #-comment banner both contain it verbatim). Kept in sync with LICENSE.tmpl.
MARKER = "SPDX-License-Identifier: Apache-2.0"


def has_header(path: Path) -> bool:
    """True if the license marker appears in the file.

    The whole file is scanned rather than only its first N bytes: the header sits
    after the module docstring / ``from __future__`` import, and a long docstring
    can push it well past any fixed byte window (which produced false positives).
    Source files are small, so scanning in full is cheap and unambiguous.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        # Unreadable (e.g. a broken symlink) -- surface it as missing so it is
        # not silently treated as compliant.
        return False
    return MARKER in text


def main(argv: list[str]) -> int:
    """Check each given path; return non-zero (and list them) if any lack the header."""
    missing = [f for f in argv if Path(f).is_file() and not has_header(Path(f))]
    if missing:
        print("License header missing from the following files:", file=sys.stderr)
        for f in sorted(missing):
            print(f"  {f}", file=sys.stderr)
        print(
            "\nAdd the header from LICENSE.tmpl. For Python put it in a module-level\n"
            '__copyright__ = """..."""  after the docstring and any '
            "`from __future__ import ...`;\n"
            "for shell/Dockerfile use the #-comment banner (after the shebang).",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
