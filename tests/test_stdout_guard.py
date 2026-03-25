"""Tests for stdout_guard module.

The critical invariant: importing lazy_claude must produce zero bytes on
the real stdout fd (fd 1). All diagnostic output must go to stderr.
"""

import os
import subprocess
import sys
import tempfile


def test_import_produces_no_stdout():
    """Importing lazy_claude must emit nothing on the real stdout fd."""
    with tempfile.TemporaryFile() as capture:
        result = subprocess.run(
            [sys.executable, "-c", "import lazy_claude"],
            stdout=capture,
            stderr=subprocess.DEVNULL,
            cwd="/Users/sonwork/Workspace/lazy-claude",
        )
        capture.seek(0)
        output = capture.read()
    assert result.returncode == 0, "import lazy_claude must not crash"
    assert output == b"", f"Expected no stdout, got: {output!r}"


def test_get_mcp_stdout_returns_writable_fd():
    """get_mcp_stdout() must return a file object writable to the real stdout fd."""
    # We test this in a subprocess to avoid polluting the test runner's stdout.
    code = """
import lazy_claude.stdout_guard as sg
fd = sg.get_mcp_stdout()
# Must be writable
fd.write(b"ok")
fd.flush()
"""
    with tempfile.TemporaryFile() as capture:
        result = subprocess.run(
            [sys.executable, "-c", code],
            stdout=capture,
            stderr=subprocess.DEVNULL,
            cwd="/Users/sonwork/Workspace/lazy-claude",
        )
        capture.seek(0)
        output = capture.read()
    assert result.returncode == 0, "get_mcp_stdout() must not crash"
    assert output == b"ok", f"Expected b'ok' on stdout, got: {output!r}"


def test_sys_stdout_goes_to_stderr_after_import():
    """After import, sys.stdout writes must reach stderr, not the real stdout fd."""
    code = """
import lazy_claude  # activates guard
import sys
sys.stdout.write("should_be_on_stderr\\n")
sys.stdout.flush()
"""
    with tempfile.TemporaryFile() as real_stdout_capture:
        with tempfile.TemporaryFile() as stderr_capture:
            result = subprocess.run(
                [sys.executable, "-c", code],
                stdout=real_stdout_capture,
                stderr=stderr_capture,
                cwd="/Users/sonwork/Workspace/lazy-claude",
            )
            real_stdout_capture.seek(0)
            real_out = real_stdout_capture.read()
            stderr_capture.seek(0)
            real_err = stderr_capture.read()

    assert result.returncode == 0
    assert real_out == b"", f"Nothing should appear on real stdout, got: {real_out!r}"
    assert b"should_be_on_stderr" in real_err, (
        f"Expected text on stderr, got: {real_err!r}"
    )
