"""stdout_guard.py — Redirect C-level fd 1 away from the MCP protocol channel.

MCP stdio transport uses the process's real stdout (fd 1) to send JSON-RPC
messages.  Any stray print() or library debug output written to fd 1 corrupts
the protocol stream, causing the host to drop the connection.

This module must be imported **before** any other module that might write to
stdout.  On import it:

1. Duplicates fd 1 into a private fd so we never lose the real stdout.
2. Redirects the C-level fd 1 to stderr so that any native library that
   writes to fd 1 directly (e.g. whisper.cpp, onnxruntime) ends up on stderr.
3. Replaces sys.stdout with a text wrapper around stderr so that Python-level
   print() and sys.stdout.write() also go to stderr.

The saved real stdout fd is exposed via get_mcp_stdout() for the MCP transport
layer to use exclusively.
"""

from __future__ import annotations

import os
import sys

# ---------------------------------------------------------------------------
# One-time setup — runs at import time.
# ---------------------------------------------------------------------------

# Step 1: duplicate the real stdout fd so we keep it alive.
_real_stdout_fd: int = os.dup(1)

# Step 2: redirect the C-level fd 1 to stderr (fd 2).
os.dup2(2, 1)

# Step 3: replace Python's sys.stdout with a stderr wrapper so print() etc.
# also go to stderr instead of attempting to write to the now-redirected fd 1.
sys.stdout = sys.stderr


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_mcp_stdout() -> "os.FileIO":
    """Return a binary file object backed by the real stdout fd.

    This is the *only* place that should write to the original stdout.
    The MCP transport layer calls this to obtain the channel it uses for
    JSON-RPC responses.
    """
    return os.fdopen(_real_stdout_fd, "wb", buffering=0, closefd=False)
