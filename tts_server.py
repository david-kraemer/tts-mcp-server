"""TTS MCP Server for Claude Code notifications.

Provides speech synthesis via MLX-audio and Kokoro-82M on Apple Silicon.
Exposes a ``notify`` tool for task completion alerts and a ``speak`` tool
for general-purpose TTS with voice/speed control.
"""

import argparse
import asyncio
import functools
import itertools
import logging
import pathlib
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field as dataclass_field

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
HUGGINGFACE_REPO = "mlx-community/Kokoro-82M-bf16"


# ---------------------------------------------------------------------------
# Playback queue — serializes audio output from concurrent tool calls
# ---------------------------------------------------------------------------


@dataclass(frozen=True, order=True)
class PlaybackItem:
    """Priority queue entry. Lower priority = more urgent (heapq convention)."""

    priority: int
    seq: int
    audio: mx.array = dataclass_field(compare=False)


class PlaybackQueue:
    """Async priority queue with a background worker that plays audio sequentially."""

    def __init__(self) -> None:
        self._counter = itertools.count()
        self._queue: asyncio.PriorityQueue[PlaybackItem] = asyncio.PriorityQueue()
        self._worker: asyncio.Task[None] | None = None

    def start(self) -> None:
        """Start the background drain task."""
        self._worker = asyncio.create_task(self._drain())
        logger.info("Playback worker started.")

    async def stop(self) -> None:
        """Cancel the worker and clean up."""
        if self._worker is not None:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
            self._worker = None
        logger.info("Playback worker stopped.")

    def enqueue(self, audio: mx.array, priority: int = 10) -> None:
        """Add audio to the queue. Non-blocking."""
        self._queue.put_nowait(
            PlaybackItem(priority, next(self._counter), audio)
        )

    async def _drain(self) -> None:
        """Pull items and play them one at a time."""
        while True:
            item = await self._queue.get()
            try:
                await _play(item.audio)
            except Exception:
                logger.exception("Playback failed")
            finally:
                self._queue.task_done()


async def _play(audio: mx.array) -> None:
    """Write audio to a temp WAV and play via afplay."""
    path = pathlib.Path(tempfile.mktemp(suffix=".wav", prefix="tts_"))
    try:
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
    finally:
        path.unlink(missing_ok=True)


_playback: PlaybackQueue | None = None


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(_server: FastMCP) -> AsyncIterator[dict]:
    """Start/stop the playback worker with the server lifecycle."""
    global _playback
    _playback = PlaybackQueue()
    _playback.start()
    try:
        yield {}
    finally:
        await _playback.stop()
        _playback = None


mcp = FastMCP("TTS Notification Server", lifespan=lifespan)


@mcp.tool()
async def notify(message: str, speed: float = SPEED) -> str:
    """Speak a short task-completion notification.

    :param message: Notification text (e.g. "Build finished").
    """
    _validate_speed(speed)
    audio = generate(message, speed=speed)
    _playback.enqueue(audio)
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
    _playback.enqueue(audio)
    dur = len(audio) / SAMPLE_RATE
    return f"Spoke {len(text)} chars in {dur:.1f}s (voice={voice}, speed={speed}x)"


@functools.lru_cache
def load_model(path: pathlib.Path = HUGGINGFACE_REPO) -> Module:
    logger.info("Loading model %s ...", path)
    model = mlx_load_model(path)
    # Warmup: compile Metal shaders so the first real call is fast.
    list(model.generate("warmup"))
    logger.info("Model loaded and warmed up.")
    return model


def generate(text: str, voice: str = DEFAULT_VOICE, speed: float = SPEED) -> mx.array:
    """Run TTS inference, return raw audio array."""
    model = load_model()
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


def main():
    logger.info("Starting TTS MCP server ...")
    # Start the MCP transport immediately so the handshake succeeds,
    # then warm up the model lazily on first tool call.
    mcp.run(transport="stdio")


def init():
    load_model()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TTS MCP server for Claude Code — Kokoro-82M on Apple Silicon.",
        epilog="Run without arguments to start the MCP server.",
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=["init"],
        default=None,
        help="'init' to pre-download the TTS model (~200 MB).",
    )
    args = parser.parse_args()

    if args.command == "init":
        logger.info("Preloading model for faster first response...")
        init()
    else:
        main()
