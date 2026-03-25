"""__main__.py — Entry point for `python -m lazy_claude` and the CLI script.

stdout_guard must be imported first (before any other lazy_claude module)
so that fd 1 is safely duplicated before any native library can write to it.
"""

from __future__ import annotations

# Import stdout_guard first — this redirects fd 1 away from native libs.
import lazy_claude.stdout_guard  # noqa: F401  (side-effect import)


def main() -> None:
    """Start the Lazy Claude MCP server."""
    from lazy_claude.server import run_server
    run_server()


if __name__ == "__main__":
    main()
