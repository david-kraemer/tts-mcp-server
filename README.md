# tts-mcp-server

Local text-to-speech for Claude Code via [MCP](https://modelcontextprotocol.io/).
Runs [Kokoro-82M](https://huggingface.co/mlx-community/Kokoro-82M-bf16) on Apple
Silicon through [MLX-audio](https://github.com/ml-explore/mlx-audio), giving Claude
the ability to speak task notifications and arbitrary text aloud.

**Requirements:** macOS on Apple Silicon (M1+), Python 3.12+.

## Setup

```bash
cd ~/projects/tts-mcp-server
uv venv && source .venv/bin/activate
uv pip install -e .

# Kokoro's G2P (misaki) needs a spacy language model.
# spacy tries to `pip install` it at runtime, which hangs in uv-managed venvs.
# Install it explicitly:
uv pip install en_core_web_sm@https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl

# Pre-download the TTS model (~200 MB, one-time):
python -c "from mlx_audio.tts.utils import load_model; load_model('mlx-community/Kokoro-82M-bf16')"
```

Register with Claude Code (user-wide, available in all projects):

```bash
claude mcp add --transport stdio --scope user tts -- \
  /path/to/tts-mcp-server/.venv/bin/python \
  /path/to/tts-mcp-server/tts_server.py
```

Verify:

```bash
claude mcp list   # should show tts: ✓ Connected
```

## Tools

### `notify(message)`

Quick task-completion alert. Speaks at 1.2x speed with the default voice
(`af_heart`). Designed for short messages like "Build finished" or "Tests
passed".

### `speak(text, voice?, speed?)`

Full TTS with voice and speed control.

| Parameter | Default     | Range / Options |
|-----------|-------------|-----------------|
| `text`    | *(required)* | Any string |
| `voice`   | `af_heart`  | See [voices](#voices) |
| `speed`   | `1.0`       | 0.5 -- 2.0 |

## Voices

Kokoro ships 54 presets. A useful subset:

| ID          | Description       |
|-------------|-------------------|
| `af_heart`  | American female (default) |
| `af_bella`  | American female   |
| `af_nova`   | American female   |
| `am_adam`   | American male     |
| `am_echo`   | American male     |
| `bf_emma`   | British female    |
| `bm_george` | British male      |

Full list: prefix `af_` / `am_` (American), `bf_` / `bm_` (British),
`jf_` / `jm_` (Japanese), `zf_` / `zm_` (Chinese).

## Architecture

```
Claude Code  ──stdio──>  FastMCP server  ──>  MLX-audio/Kokoro  ──>  afplay
```

- **Lazy loading:** The model, spacy G2P pipeline, and Metal shaders all
  initialize on the first tool call (~6 s). This keeps the MCP handshake
  instant so health checks pass. Subsequent calls run in ~0.1 s.
- **No persistent daemon:** Claude Code spawns the server on session start and
  kills it on exit. No LaunchAgent needed.
- **Temp files:** Audio is written to a temp WAV, played with `afplay`, then
  deleted. No disk accumulation.

## Performance (M2 Max)

| Metric | First call | Subsequent |
|--------|-----------|------------|
| Latency (short phrase) | ~6 s | ~0.1 s |
| Memory | ~420 MB | ~420 MB |
| CPU | < 5% (GPU-accelerated) | < 5% |

## Troubleshooting

**Server shows `✗ Failed to connect`:** The model is loading during the health
check. This was fixed by deferring model load to first tool call. If you still
see this, ensure you're using the version of `tts_server.py` that calls
`mcp.run()` immediately in `__main__`.

**First call is slow (~6 s):** Expected. Spacy G2P pipeline and Metal shader
compilation happen once per server lifetime. After that, calls are sub-200 ms.

**First call hangs indefinitely:** The spacy `en_core_web_sm` model is missing.
Misaki's G2P calls `spacy.cli.download()` at runtime, which shells out to `pip`
-- but uv-managed venvs don't have `pip`, so it hangs forever. Fix:
```bash
uv pip install en_core_web_sm@https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
```

**`afplay` not found:** You're not on macOS. Replace the `afplay` call in
`_play()` with your platform's audio player (e.g., `paplay` on Linux, `sox`
cross-platform).

**Model download fails:** Pre-download manually:
```bash
pip install huggingface-hub
huggingface-cli download mlx-community/Kokoro-82M-bf16
```

## License

MIT
