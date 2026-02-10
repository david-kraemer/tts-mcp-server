"""TTS MCP Server for Claude Code notifications.

Provides speech synthesis via MLX-audio and Kokoro-82M on Apple Silicon.
Exposes a ``notify`` tool for task completion alerts and a ``speak`` tool
for general-purpose TTS with voice/speed control.
"""

import asyncio
import functools
import logging
import pathlib
import tempfile

import mlx.core as mx
import soundfile as sf
from fastmcp import FastMCP
from mlx.nn.layers import Module
from mlx_audio.tts.utils import load_model as mlx_load_model
from rich.logging import RichHandler

logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])
logger = logging.getLogger(__name__)


# NOTE: ATM highest quality Kokoro voice, cf. https://tinyurl.com/32hcr3b7
DEFAULT_VOICE = "af_heart"
SAMPLE_RATE = 24000
SPEED = 1.2
HUGGINGFACE_REPO = pathlib.Path("mlx-community/Kokoro-82M-bf16")


mcp = FastMCP("TTS Notification Server")


@mcp.tool()
async def notify(message: str, speed: float = SPEED) -> str:
    """Speak a short task-completion notification.

    :param message: Notification text (e.g. "Build finished").
    """
    _validate_speed(speed)
    audio = generate(message, speed=speed)
    await play(audio)
    return f"Notified: {message}"


@mcp.tool()
async def speak(text: str, voice: str = DEFAULT_VOICE, speed: float = SPEED) -> str:
    """Generate and play speech with voice and speed control.

    :param text: Text to speak.
    :param voice: Kokoro voice preset
    :param speed: Playback speed multiplier, 0.5–2.0.
    """
    _validate_speed(speed)
    audio = generate(text, voice=voice, speed=speed)
    await play(audio)
    dur = len(audio) / SAMPLE_RATE
    return f"Spoke {len(text)} chars in {dur:.1f}s (voice={voice}, speed={speed}x)"


async def play(audio: mx.array) -> None:
    """Write audio to a temp WAV and play via afplay."""
    path = temp_dir() / "out.wav"
    sf.write(str(path), audio, SAMPLE_RATE)
    proc = await asyncio.create_subprocess_exec(
        "afplay",
        str(path),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"afplay failed: {stderr.decode()}")
    path.unlink(missing_ok=True)


@functools.lru_cache
def load_model(path: pathlib.Path = HUGGINGFACE_REPO) -> Module:
    logger.info("Loading model %s ...", path)
    model = mlx_load_model(path)
    # Warmup: compile Metal shaders so the first real call is fast.
    list(model.generate("warmup", voice=DEFAULT_VOICE))
    logger.info("Model loaded and warmed up.")
    return model


@functools.lru_cache
def temp_dir() -> pathlib.Path:
    """Get the temp directory path, creating it if necessary."""
    _temp_dir = pathlib.Path(tempfile.mkdtemp(prefix="tts_"))
    return _temp_dir


def generate(text: str, voice: str = DEFAULT_VOICE, speed: float = SPEED) -> mx.array:
    """Run TTS inference, return raw audio array."""
    model = load_model(voice=voice)
    chunks = [
        r.audio
        for r in model.generate(text=text, voice=voice, speed=speed, lang_code="a")
    ]
    if not chunks:
        raise RuntimeError("No audio generated")
    return mx.concatenate(chunks) if len(chunks) > 1 else chunks[0]


def _validate_speed(speed: float) -> None:
    if not 0.5 <= speed <= 2.0:
        raise ValueError(f"Speed must be 0.5–2.0, got {speed}")


if __name__ == "__main__":
    logger.info("Starting TTS MCP server ...")
    # Start the MCP transport immediately so the handshake succeeds,
    # then warm up the model lazily on first tool call.
    mcp.run(transport="stdio")
