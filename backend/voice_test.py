"""Utility script for testing OpenAI Realtime transcription with microphone input.

Run this script to stream audio from your default microphone to the OpenAI
Realtime transcription endpoint and print transcription events as they arrive.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import queue
import signal
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncGenerator

from dotenv import load_dotenv
import sounddevice as sd
import websockets

load_dotenv()

REALTIME_URL = "wss://api.openai.com/v1/realtime?intent=transcription"
SAMPLE_RATE = 16_000
CHANNELS = 1
# Frames per audio chunk; adjust to tune latency vs. network overhead (20 ms at 16 kHz).
FRAMES_PER_CHUNK = int(SAMPLE_RATE * 0.02)


@dataclass
class TranscriptionConfig:
    """Holds runtime configuration values for the Realtime session."""

    prompt: str = "This is a customer support call about technical issues."
    language: str = "en"
    vad_threshold: float = 0.5
    vad_prefix_padding_ms: int = 500
    vad_silence_duration_ms: int = 2000


async def get_microphone_audio(shutdown_event: asyncio.Event) -> AsyncGenerator[bytes, None]:
    """Yield raw PCM16 audio chunks from the system microphone until shutdown_event is set."""

    audio_queue: queue.Queue[bytes] = queue.Queue()

    def _callback(indata, frames, _time, status):
        if status:
            print(f"[sounddevice] {status}", file=sys.stderr)
        audio_queue.put(bytes(indata))

    stream = sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=FRAMES_PER_CHUNK,
        dtype="int16",
        channels=CHANNELS,
        callback=_callback,
    )

    with stream:
        while not shutdown_event.is_set():
            try:
                chunk = await asyncio.to_thread(audio_queue.get, True, 0.1)
            except queue.Empty:
                continue
            if chunk:
                yield chunk


async def stream_microphone_audio(
    websocket: websockets.WebSocketClientProtocol,
    shutdown_event: asyncio.Event,
) -> None:
    """Stream microphone audio to the realtime endpoint until shutdown_event is set."""

    try:
        async for audio_chunk in get_microphone_audio(shutdown_event):
            audio_b64 = base64.b64encode(audio_chunk).decode("ascii")
            payload = {
                "type": "input_audio_buffer.append",
                "audio": audio_b64,
            }
            await websocket.send(json.dumps(payload))
            if shutdown_event.is_set():
                break
    finally:
        shutdown_event.set()


def build_session_update(config: TranscriptionConfig) -> str:
    """Create the session.update payload for configuring transcription."""

    session_update = {
        "type": "transcription_session.update",
        "session": {
            "input_audio_format": "pcm16",
            "input_audio_transcription": {
                "model": "gpt-4o-transcribe",
                "prompt": config.prompt,
                "language": config.language,
            },
            "turn_detection": {
                "type": "server_vad",
                "threshold": config.vad_threshold,
                "prefix_padding_ms": config.vad_prefix_padding_ms,
                "silence_duration_ms": config.vad_silence_duration_ms,
            },
        },
    }
    return json.dumps(session_update)


@asynccontextmanager
async def realtime_connection(api_key: str):
    """Async context manager that yields an authenticated websocket connection."""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "OpenAI-Beta": "realtime=v1",
    }
    websocket = await websockets.connect(REALTIME_URL, additional_headers=headers)
    try:
        yield websocket
    finally:
        await websocket.close()


async def transcription_agent(config: TranscriptionConfig) -> None:
    """Run a continuous transcription session and print realtime events."""

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set.")

    shutdown_event = asyncio.Event()

    loop = asyncio.get_running_loop()

    def _handle_signal(signum, _frame):
        print(f"\nReceived signal {signum}; stopping audio stream.")
        shutdown_event.set()

    # Ensure Ctrl+C stops the audio task promptly.
    for signame in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signame, _handle_signal, signame, None)
        except NotImplementedError:
            # add_signal_handler is not available on Windows event loops.
            signal.signal(signame, lambda _sig, _frm: shutdown_event.set())

    async with realtime_connection(api_key) as websocket:
        # Configure the session for transcription-only use.
        await websocket.send(build_session_update(config))

        audio_task = asyncio.create_task(
            stream_microphone_audio(websocket, shutdown_event)
        )

        try:
            while not shutdown_event.is_set():
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=0.25)
                except asyncio.TimeoutError:
                    continue
                except websockets.exceptions.ConnectionClosed:
                    break

                event = json.loads(message)
                event_type = event.get("type")
                if not event_type:
                    continue

                # Print events verbosely for debugging / observability.

                if event_type == "conversation.item.input_audio_transcription.completed":
                    transcript = event.get("transcript")
                    if transcript:
                        print(transcript)
                    continue

                # elif event_type == "conversation.item.created":
                #     item = event.get("item") or {}
                #     if item.get("type") == "transcript":
                #         contents = item.get("content") or []
                #         for content in contents:
                #             if content.get("type") == "input_text" and content.get("text"):
                #                 print(f"\nTranscript item: {content['text']}\n")
                #     else:
                #         print("Conversation item payload:")
                #         print(json.dumps(item, indent=2))

                elif event_type == "error":
                    print("Error detail:")
                    print(json.dumps(event, indent=2))
        finally:
            shutdown_event.set()
            try:
                await websocket.close()
            except websockets.exceptions.ConnectionClosed:
                pass
            try:
                await audio_task
            except websockets.exceptions.ConnectionClosed:
                pass


async def main() -> None:
    config = TranscriptionConfig()
    await transcription_agent(config)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Interrupted by user")
