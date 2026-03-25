"""server.py — FastMCP server exposing voice I/O tools.

Transport: stdio, using the real stdout fd from stdout_guard so that
native libraries (whisper.cpp, onnxruntime, torch) can never pollute
the MCP protocol channel.

Tools exposed
-------------
ask_user_voice(questions: list[str]) -> str
    Half-duplex voice Q&A: speak each question, record answer, transcribe.

speak_message(text: str) -> dict
    TTS-only output.  Returns {"status": "spoken", "chars": len(text)}.

toggle_listening(enabled: bool) -> dict
    Enable/disable microphone recording.  When disabled ask_user_voice
    still speaks the question but skips recording.
    Returns {"listening": enabled}.
"""

from __future__ import annotations

import sys
import threading
from io import TextIOWrapper
from typing import Any

# stdout_guard MUST be imported first so that the real stdout fd is
# preserved before anything else writes to it.
from lazy_claude.stdout_guard import get_mcp_stdout  # noqa: E402 (intentional early import)

# All logging goes to stderr.
def _log(msg: str) -> None:
    print(f"[lazy-claude server] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Lazy imports for heavy dependencies
# ---------------------------------------------------------------------------

from lazy_claude.tts import TTSEngine
from lazy_claude.stt import load_model, transcribe
from lazy_claude.audio import load_vad_model, record_audio


# ---------------------------------------------------------------------------
# VoiceServer — state holder + tool implementations
# ---------------------------------------------------------------------------

class VoiceServer:
    """Holds shared state and implements the voice tool logic.

    Attributes
    ----------
    listening : bool
        When False, ask_user_voice skips mic recording and returns a
        "skipped — listening paused" placeholder.
    busy : bool
        True while a voice turn is in progress.  Concurrent calls are
        rejected immediately.
    tts : TTSEngine
        Shared TTS engine instance.
    """

    def __init__(self) -> None:
        _log("Initialising VoiceServer…")
        self.tts = TTSEngine()
        self._whisper_model = load_model()
        self._vad_model = load_vad_model()

        self.listening: bool = True
        self.busy: bool = False
        self._lock = threading.Lock()
        _log("VoiceServer ready.")

    # ------------------------------------------------------------------
    # Tool implementations (plain Python — called by MCP tool wrappers)
    # ------------------------------------------------------------------

    def toggle_listening_impl(self, *, enabled: bool) -> dict[str, Any]:
        """Enable or disable microphone recording."""
        self.listening = enabled
        _log(f"Listening {'enabled' if enabled else 'disabled'}.")
        return {"listening": enabled}

    def speak_message_impl(self, *, text: str) -> dict[str, Any]:
        """Speak text via TTS and return a status dict."""
        _log(f"speak_message: {len(text)} chars")
        try:
            self.tts.speak(text)
        except Exception as exc:
            _log(f"WARNING: TTS error during speak_message: {exc}")
        return {"status": "spoken", "chars": len(text)}

    def ask_user_voice_impl(self, *, questions: list[str]) -> str:
        """Speak each question, record and transcribe each answer.

        Returns a newline-separated string of "Q: …\\nA: …" blocks.
        """
        # Concurrent call protection
        with self._lock:
            if self.busy:
                _log("ask_user_voice: rejected — already busy")
                return "A: (busy — already processing a voice turn)"
            self.busy = True

        try:
            return self._run_qa_session(questions)
        finally:
            with self._lock:
                self.busy = False

    def _run_qa_session(self, questions: list[str]) -> str:
        parts: list[str] = []
        for question in questions:
            qa = self._ask_single(question)
            parts.append(qa)
        return "\n\n".join(parts)

    def _ask_single(self, question: str) -> str:
        """Speak one question, record response, transcribe, return QA block."""
        # Step 1 + 2: speak question and wait for TTS to finish
        _log(f"Speaking question: {question!r}")
        try:
            self.tts.speak(question)
        except Exception as exc:
            _log(f"WARNING: TTS error while speaking question: {exc}")

        # Step 3 + 4: listening disabled path — skip mic
        if not self.listening:
            _log("Listening disabled — skipping mic recording.")
            return f"Q: {question}\nA: (skipped — listening paused)"

        # Step 3: record audio via VAD
        _log("Recording answer…")
        try:
            audio = record_audio(vad_model=self._vad_model)
        except RuntimeError as exc:
            _log(f"ERROR: mic capture failed: {exc}")
            return f"Q: {question}\nA: (error — mic failed: {exc})"
        except Exception as exc:
            _log(f"ERROR: unexpected mic error: {exc}")
            return f"Q: {question}\nA: (error — {exc})"

        # Step 4: handle timeout
        if audio is None:
            _log("No speech detected — timed out.")
            return f"Q: {question}\nA: (no response — timed out)"

        # Step 5: transcribe
        _log("Transcribing…")
        answer = transcribe(audio, model=self._whisper_model)

        return f"Q: {question}\nA: {answer}"

    def shutdown(self) -> None:
        """Release resources on server stop."""
        _log("VoiceServer shutting down…")
        try:
            self.tts.stop()
        except Exception:
            pass
        _log("VoiceServer shutdown complete.")


# ---------------------------------------------------------------------------
# FastMCP app factory
# ---------------------------------------------------------------------------

def create_server() -> "mcp.server.fastmcp.FastMCP":  # type: ignore[name-defined]
    """Build and return the FastMCP app with all voice tools registered."""
    from mcp.server.fastmcp import FastMCP

    voice = VoiceServer()
    app = FastMCP(
        "lazy-claude",
        log_level="WARNING",  # keep FastMCP internal logs off stderr noise
    )

    @app.tool()
    def ask_user_voice(questions: list[str]) -> str:
        """Ask the user one or more questions via voice (TTS + mic + STT).

        For each question: speaks it aloud, waits for TTS to finish, records
        the user's spoken answer with VAD, then transcribes with Whisper.

        Returns a formatted string with Q/A pairs, one per question.
        """
        return voice.ask_user_voice_impl(questions=questions)

    @app.tool()
    def speak_message(text: str) -> dict:
        """Speak a message aloud via TTS without recording.

        Returns {"status": "spoken", "chars": <number of characters spoken>}.
        """
        return voice.speak_message_impl(text=text)

    @app.tool()
    def toggle_listening(enabled: bool) -> dict:
        """Enable or disable microphone recording.

        When disabled, ask_user_voice will still speak the question but skip
        recording and return a "(skipped — listening paused)" answer.
        Call toggle_listening(true) to re-enable.

        Returns {"listening": <current state>}.
        """
        return voice.toggle_listening_impl(enabled=enabled)

    return app


# ---------------------------------------------------------------------------
# run_server — wire up stdio transport with guarded stdout fd
# ---------------------------------------------------------------------------

def run_server() -> None:
    """Start the MCP server on stdio, using the preserved real stdout fd."""
    import anyio
    from mcp.server.stdio import stdio_server

    app = create_server()

    # Build a text-mode wrapper around the *real* stdout fd (not the
    # redirected one — stdout_guard already redirected fd 1 to stderr).
    real_stdout_binary = get_mcp_stdout()
    real_stdout_text = TextIOWrapper(real_stdout_binary, encoding="utf-8", line_buffering=True)

    async def _run() -> None:
        async with stdio_server(stdout=anyio.wrap_file(real_stdout_text)) as (
            read_stream,
            write_stream,
        ):
            await app._mcp_server.run(
                read_stream,
                write_stream,
                app._mcp_server.create_initialization_options(),
            )

    _log("Starting lazy-claude MCP server (stdio transport)…")
    try:
        anyio.run(_run)
    except KeyboardInterrupt:
        _log("Server interrupted.")
    finally:
        _log("Server stopped.")
