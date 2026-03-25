# claude-voice: Voice MCP Server

A Model Context Protocol (MCP) server that provides voice input/output integration for MCP clients such as Claude Code and Codex, enabling hands-free interaction and voice-driven workflows on macOS.

## Quick Install

### 1. Install system dependencies on macOS
```bash
brew install espeak-ng ffmpeg
```

### 2. Register the MCP server in Claude Code
```bash
claude mcp add claude-voice npx @tuan_son.dinh/claude-voice
```

This is the main install path. No repo clone is required.

Or add manually to `~/.claude.json` under `mcpServers`:
```json
{
  "claude-voice": {
    "command": "npx",
    "args": ["@tuan_son.dinh/claude-voice"]
  }
}
```

> On first run the server auto-installs its Python dependencies (~500MB). Subsequent starts are instant.

### 3. Restart Claude Code

If Claude Code is already running, restart it so the new MCP server is loaded.

That is enough for installation. No `CLAUDE.md` changes are required just to make the MCP server available.

### Optional: Global Voice-Mode Prompt

The MCP server works without any prompt-file changes, but users will not get the same hands-free workflow unless their global prompt tells the agent when to enter voice mode and how to stay there.

If you want the same experience, add a section like this to your global `CLAUDE.md`:

```markdown
## Voice I/O
- `claude-voice` MCP provides voice input/output integration for hands-free workflows
- Tool: `mcp__claude-voice__ask_user_voice` speaks questions aloud and records voice replies
- Tool: `mcp__claude-voice__speak_message` speaks short confirmations or updates

### Voice Mode Toggle
- Voice mode is OFF by default
- Voice mode is activated when your preferred trigger fires, for example a keybinding or explicit user request
- On activation, call `mcp__claude-voice__ask_user_voice` with: "Now in voice mode. What would you like to do?"
- Stay in voice mode for the rest of the conversation until the user explicitly exits it

### Voice Mode Rules
- While in voice mode, never ask for typed replies
- Use `mcp__claude-voice__ask_user_voice` for all follow-up questions
- Use `mcp__claude-voice__speak_message` for short spoken confirmations when useful
- Exit voice mode only when the user says something like "exit voice mode", "stop voice mode", or "back to text"
- Do not leave voice mode because of errors or tool failures; keep interacting by voice unless the user explicitly switches back to text
```

This prompt setup is recommended if you want the full voice-mode behavior. MCP installation alone only makes the tools available.

## How To Use

Once the MCP server is installed, there are two common ways to start using it:

### 1. Ask Claude to enable voice mode

If your prompt setup includes voice-mode instructions, tell Claude something like:

```text
Enable voice mode
```

or:

```text
Switch to voice mode
```

Claude should then start using `ask_user_voice` for spoken interaction instead of asking for typed replies.

### 2. Trigger voice mode with a hook or keybinding

If you have a local hook configured, you can bind a key such as `V` to trigger voice mode.

Typical behavior:
- press `V`
- your hook tells Claude to enter voice mode
- Claude calls `ask_user_voice` with an initial spoken prompt such as: "Now in voice mode. What would you like to do?"

This is optional client-side setup. The MCP server does not create the `V` keybinding by itself; your local Claude workflow or hook needs to do that.

### Leaving voice mode

If your prompt setup mirrors the example in this README, exit by saying something like:
- `exit voice mode`
- `stop voice mode`
- `back to text`

### Other MCP Clients

Any stdio MCP client can run this server. For local development, the generic command is:

```json
{
  "claude-voice": {
    "command": "uv",
    "args": ["--directory", "/absolute/path/to/claude-voice", "run", "python", "-m", "lazy_claude"]
  }
}
```

Use your MCP client's normal server-registration flow and substitute the correct local path.

---

## Features

- **Voice Input**: Speak questions and receive transcribed responses automatically submitted
- **Text-to-Speech**: Convert text responses to spoken audio with natural-sounding output
- **Real-Time Recording**: Stream audio input with automatic silence detection and submission
- **Fast Model Loading**: Efficient ML model management for speech recognition and synthesis
- **Fully Integrated**: Registered as `claude-voice` MCP server in Claude Code

## Prerequisites

### System Requirements
- **macOS** (Apple Silicon or Intel) or Linux/Windows with fallback
- **Python** 3.12+
- **System Tools**:
  ```bash
  brew install espeak-ng ffmpeg
  ```

### Dependencies
Automatically installed via `pyproject.toml`:
- `kokoro` — High-quality text-to-speech engine
- `pywhispercpp` — Fast speech-to-text (Whisper.cpp binding)
- `sounddevice` — Audio input/output (fallback path)
- `soundfile` — Audio file handling
- `onnxruntime` — ML inference for TTS and VAD
- `mcp[cli]` — Model Context Protocol server framework
- `numpy` — Numerical computing

**macOS-specific (optional, for system AEC)**:
- `pyobjc-framework-AVFoundation` — AVAudioEngine bindings
- `pyobjc-framework-Foundation` — Device change notifications

Install with macOS support:
```bash
pip install -e '.[macos]'
# or
uv sync --all-extras
```

## Setup

### 1. Clone or Navigate to Project
```bash
cd /Users/sonwork/Workspace/claude-voice
```

### 2. Install System Dependencies
```bash
brew install espeak-ng ffmpeg
```

### 3. Set Up Python Environment
```bash
# Sync dependencies using uv
uv sync

# Or use pip
pip install -e .
```

### 4. Register with Your MCP Client

#### Claude Code
The server can be registered in `~/.claude.json`:
```json
{
  "claude-voice": {
    "command": "uv",
    "args": ["--directory", "/Users/sonwork/Workspace/claude-voice", "run", "python", "-m", "lazy_claude"]
  }
}
```

If not present, add this entry to the `mcpServers` object in `~/.claude.json`.

#### Codex and Other MCP Clients

Register the same stdio command in your client's MCP config:

```json
{
  "claude-voice": {
    "command": "uv",
    "args": ["--directory", "/absolute/path/to/claude-voice", "run", "python", "-m", "lazy_claude"]
  }
}
```

No `CLAUDE.md` update is required. If your client supports project instructions, you can optionally add the short voice-tool guidance shown above.

## Usage

### Starting the Server
The server runs automatically when Claude Code loads the `claude-voice` MCP. No manual startup needed.

### Available Tools

### Wake Word Setup

Wake-word mode now uses `openWakeWord`, so no Picovoice account or access key is needed.

Install it in the active environment:
```bash
pip install openwakeword
```

Optional environment variables:
```bash
export OPENWAKEWORD_MODEL_PATH=/absolute/path/to/model.onnx
export OPENWAKEWORD_THRESHOLD=0.5
export OPENWAKEWORD_VAD_THRESHOLD=0.5
```

If `OPENWAKEWORD_MODEL_PATH` is unset, `openWakeWord` will use its default bundled/pretrained model configuration.

#### `ask_user_voice(questions: list[str]) → str`
Reads questions aloud and records voice replies with automatic submission.

**Example**:
```python
result = ask_user_voice([
    "What is your name?",
    "What task should I help you with?"
])
# Output: "Q: What is your name?\nA: John\nQ: What task should I help you with?\nA: Fix the bug in the login page"
```

**Parameters**:
- `questions` — List of questions to ask the user

**Returns**: Formatted string with Q/A pairs (one per line): `Q: question\nA: answer`

---

#### `speak_message(text: str) → dict`
Converts text to speech and plays it aloud.

**Example**:
```python
result = speak_message("Your task has been completed successfully")
# Output: {"status": "spoken", "chars": 45}
```

**Parameters**:
- `text` — Message to speak

**Returns**: `{"status": "spoken", "chars": <character count>}`

---

#### `toggle_listening(enabled: bool) → dict`
Enable or disable the voice input listening mode (useful for background noise or multi-tasking).

**Example**:
```python
result = toggle_listening(enabled=False)
# Output: {"listening": False}
```

**Parameters**:
- `enabled` — `True` to enable, `False` to disable

**Returns**: `{"listening": <bool>}` — the current listening state after the change

---

## Voice Workflow Integration

### In Claude Code
If you want Claude Code to behave like a hands-free voice assistant, add the global prompt section above. A minimal version is:

```markdown
Use the ask_user_voice tool to get the user's voice input on whether to proceed with deployment.
```

### Best Practices
1. **Ask Single Questions**: Use clear, concise questions (one per call)
2. **Batch When Appropriate**: Group related questions in a single call to reduce back-and-forth
3. **Mic Is Turn-Based**: Voice capture is activated only during `ask_user_voice(...)`, not as an always-listening input mode
4. **Handle Pauses**: The listener segments after about `1.0s` of silence, but the server keeps the reply open for up to `5s` waiting for continuation
5. **Use `OVER` To Submit**: Ending a segment with `OVER` followed by the normal `1.5s` pause submits immediately and removes `OVER` from the returned transcript
6. **Provide Context**: Speak response confirmations back using `speak_message` for feedback loops

## Architecture

The server supports **two audio backends**, automatically selected at startup:

### macOS System AEC Path (Preferred)
On macOS, the server uses `AVAudioEngine` with the Voice Processing IO audio unit
(same tech as FaceTime) for instant, hardware-level echo cancellation:

- **Single `AVAudioBackend` instance** manages both mic input and TTS output through
  one `AVAudioEngine` with voice processing enabled
- **Instant echo cancellation** — no convergence time, system AEC handles it all
- **Mic tap** at native hardware rate (44.1 or 48 kHz) → resample to 16 kHz → deliver
  512-sample chunks to VAD
- **No custom AEC code** — system provides automatic noise suppression + AGC
- **No ReferenceBuffer** — system AEC sees the output automatically
- **Device change recovery** — engine restarts on speaker/headphone/Bluetooth switches

See [ARCHITECTURE.md](./ARCHITECTURE.md) for detailed macOS path flow and AVAudioBackend internals.

### Fallback Path (Non-macOS or on Failure)
When macOS AVAudioEngine is unavailable or fails, the server falls back to `sounddevice`
with a custom adaptive AEC pipeline:

1. **PBFDLMS adaptive filter** — Partitioned-block frequency-domain LMS filter that
   learns and cancels the speaker-to-microphone acoustic path in real time.
2. **Residual Echo Suppression (RES)** — Spectral subtraction post-filter that catches
   echo components the adaptive filter misses (especially during early convergence).
3. **Geigel Double-Talk Detector (DTD)** — Freezes filter adaptation when user speech
   and TTS overlap, preventing filter divergence.
4. **Fallback gate** — Suppresses chunks with high residual power during TTS playback
   when AEC has not yet converged (safety net for the first few seconds).
5. **AEC calibration chirp** — At startup, a quiet 1.5-second chirp (200–4000 Hz)
   plays to train the filter before real TTS utterances.

The mic stays open throughout TTS playback — no hard muting. Barge-in works by running
VAD on the AEC-cleaned signal; when the user speaks, VAD fires and TTS stops.

**Fallback components:**
- **`audio.py`** — ContinuousListener with sounddevice + VAD + barge-in
- **`tts.py`** — TTSEngine with sounddevice output + ReferenceBuffer for AEC
- **`aec.py`** — PBFDLMS filter + RES + DTD + device change handling

### Shared Components
- **`server.py`** — MCP server setup, path detection, tool registration
- **`av_audio.py`** — AVAudioBackend, MacOSContinuousListener, MacOSTTSEngine (macOS only)
- **`stt.py`** — Speech-to-text using Whisper.cpp (both paths)
- **`stdout_guard.py`** — Prevents audio interference with Claude Code output

### Manual AEC Test Script

To generate a before/after comparison of the echo cancellation:

```bash
# Uses a synthetic 440 Hz tone (no WAV needed):
python scripts/test_aec_manual.py

# Or pass your own WAV file:
python scripts/test_aec_manual.py /path/to/speech.wav --duration 10

# Output written to ./aec_test_output/
#   raw_mic.wav        - unprocessed microphone signal
#   cleaned_aec.wav    - after adaptive filter only
#   cleaned_res.wav    - after adaptive filter + residual echo suppression
```

### Model Downloads
Models are cached locally:
- **Whisper model** → `~/.cache/huggingface/...`
- **Kokoro TTS** → `~/.cache/kokoro/...`

First-run setup will download models (~500MB total). Subsequent runs use cached models.

## Troubleshooting

### Issue: No audio input detected
**Solution**:
1. Check microphone permissions: Settings > Security & Privacy > Microphone
2. Test audio device: `python -m sounddevice`
3. Verify `espeak-ng` and `ffmpeg`:
   ```bash
   espeak-ng --version
   ffmpeg -version
   ```

### Issue: Server fails to start
**Solution**:
1. Verify Python 3.12+: `python --version`
2. Check dependencies: `uv sync` or `pip install -e .`
3. Test manual startup: `uv run python -m lazy_claude`

### Issue: Poor speech recognition
**Solution**:
1. Speak clearly and at normal volume
2. Reduce background noise
3. Use `toggle_listening(False)` to disable during loud environments

### Issue: TTS playback issues
**Solution**:
1. Check speaker volume and mute status
2. Verify audio device: `python -m sounddevice`
3. Test playback: `speak_message("Test")`

### First-Run Model Download Warnings
Warnings about missing HF_TOKEN or CUDA are normal. Models download from Hugging Face automatically.

## Development

### Running Tests
```bash
pytest tests/ -v
```

### Running with Logging
```bash
uv run python -m lazy_claude
```

### Project Structure
```
claude-voice/
├── lazy_claude/
│   ├── __main__.py              # Entry point
│   ├── server.py                # MCP server, path detection, tool definitions
│   ├── av_audio.py              # macOS: AVAudioBackend, MacOSContinuousListener, MacOSTTSEngine
│   ├── audio.py                 # Fallback: ContinuousListener with sounddevice + VAD
│   ├── tts.py                   # Fallback: TTSEngine with sounddevice + ReferenceBuffer
│   ├── aec.py                   # Fallback: PBFDLMS + RES + DTD echo cancellation
│   ├── stt.py                   # Speech-to-text (both paths)
│   ├── stdout_guard.py          # Output safety wrapper
│   └── models/                  # Model files and utilities
├── scripts/
│   ├── test_aec_manual.py       # Manual AEC before/after comparison (fallback path)
│   └── test_macos_aec.py        # Manual macOS AEC test (system path)
├── tests/                       # Test suite (unit + integration)
├── ARCHITECTURE.md              # Detailed architecture guide
├── pyproject.toml               # Project metadata and dependencies
└── README.md                    # This file
```

## Requirements Specification

### Hardware
- Microphone and speakers (or headphones)
- Minimum 4GB RAM for model loading

### Software
- macOS 10.15+
- Python 3.12+
- espeak-ng and ffmpeg

### Network
- Internet connection (first-run model downloads)
- No ongoing network usage after model caching

## License

Part of the claude-voice project. See LICENSE for details.

## Support

For issues or feature requests, refer to the project repository or contact the maintainer.
