# claude-voice: Voice MCP Server for Claude Code

A Model Context Protocol (MCP) server that provides voice input/output integration for Claude Code, enabling hands-free interaction and voice-driven workflows on macOS.

## Quick Install

**Prerequisites** (macOS):
```bash
brew install espeak-ng ffmpeg
```

**Add to Claude Code** (one command):
```bash
claude mcp add claude-voice npx @tuan_son.dinh/claude-voice
```

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

That's it — Claude Code will download and run the server automatically on next start.

---

## Features

- **Voice Input**: Speak questions and receive transcribed responses automatically submitted
- **Text-to-Speech**: Convert text responses to spoken audio with natural-sounding output
- **Real-Time Recording**: Stream audio input with automatic silence detection and submission
- **Fast Model Loading**: Efficient ML model management for speech recognition and synthesis
- **Fully Integrated**: Registered as `claude-voice` MCP server in Claude Code

## Prerequisites

### System Requirements
- **macOS** (Apple Silicon or Intel)
- **Python** 3.12+
- **System Tools**:
  ```bash
  brew install espeak-ng ffmpeg
  ```

### Dependencies
Automatically installed via `pyproject.toml`:
- `kokoro` — High-quality text-to-speech engine
- `pywhispercpp` — Fast speech-to-text (Whisper.cpp binding)
- `sounddevice` — Audio input/output
- `soundfile` — Audio file handling
- `onnxruntime` — ML inference for TTS
- `mcp[cli]` — Model Context Protocol server framework
- `numpy` — Numerical computing

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

### 4. Register with Claude Code
The server is pre-registered in `~/.claude.json`:
```json
{
  "claude-voice": {
    "command": "uv",
    "args": ["--directory", "/Users/sonwork/Workspace/claude-voice", "run", "python", "-m", "lazy_claude"]
  }
}
```

If not present, add this entry to the `mcpServers` object in `~/.claude.json`.

## Usage

### Starting the Server
The server runs automatically when Claude Code loads the `claude-voice` MCP. No manual startup needed.

### Available Tools

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
Always use `mcp__claude-voice__ask_user_voice` for voice input when handling user interaction:

```markdown
Use the ask_user_voice tool to get the user's voice input on whether to proceed with deployment.
```

### Best Practices
1. **Ask Single Questions**: Use clear, concise questions (one per call)
2. **Batch When Appropriate**: Group related questions in a single call to reduce back-and-forth
3. **Handle Silence**: The system auto-detects silence (>1.5s) and submits responses
4. **Provide Context**: Speak response confirmations back using `speak_message` for feedback loops

## Architecture

### Core Components
- **`server.py`** — MCP server setup and tool registration
- **`stt.py`** — Speech-to-text using Whisper.cpp
- **`tts.py`** — Text-to-speech using Kokoro; pushes audio into AEC reference buffer
- **`audio.py`** — Always-on mic with VAD, barge-in detection, and device change handling
- **`aec.py`** — Acoustic Echo Cancellation (AEC) pipeline
- **`stdout_guard.py`** — Prevents audio interference with Claude Code output

### Acoustic Echo Cancellation (AEC)

The server uses a multi-stage AEC pipeline to suppress TTS echo at the microphone:

1. **PBFDLMS adaptive filter** — Partitioned-block frequency-domain LMS filter that
   learns and cancels the speaker-to-microphone acoustic path in real time.
2. **Residual Echo Suppression (RES)** — Spectral subtraction post-filter that catches
   echo components the adaptive filter misses (especially during early convergence).
3. **Geigel Double-Talk Detector (DTD)** — Freezes filter adaptation when user speech
   and TTS overlap, preventing filter divergence.
4. **Fallback gate** — Suppresses chunks with high residual power during TTS playback
   when AEC has not yet converged (safety net for the first few seconds).
5. **Device change handling** — When PortAudio reports a device change, filter
   coefficients are reset and delay estimation re-runs automatically.

The mic stays open throughout TTS playback — no hard muting. Barge-in works by running
VAD on the AEC-cleaned signal; when the user speaks, VAD fires and TTS stops.

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
│   ├── __main__.py          # Entry point
│   ├── server.py            # MCP server and tool definitions
│   ├── stt.py               # Speech-to-text implementation
│   ├── tts.py               # Text-to-speech implementation
│   ├── audio.py             # Always-on mic, VAD, barge-in, device change
│   ├── aec.py               # Acoustic Echo Cancellation (PBFDLMS + RES + DTD)
│   ├── stdout_guard.py      # Output safety wrapper
│   └── models/              # Model files and utilities
├── scripts/
│   └── test_aec_manual.py   # Manual AEC before/after comparison tool
├── tests/                   # Test suite
├── pyproject.toml           # Project metadata and dependencies
└── README.md                # This file
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
