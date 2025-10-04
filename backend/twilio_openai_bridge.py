"""Minimal Twilio media stream to OpenAI Realtime bridge.

Run with: `uvicorn twilio_openai_bridge:app --host 0.0.0.0 --port 8000`
"""

import asyncio
import base64
import json
import os
import queue
import threading
from typing import Any

import audioop
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import websocket
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-realtime-preview-2024-12-17")

if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY before starting the bridge.")

app = FastAPI()


class OpenAIRealtimeClient:
    """Very small helper around websocket-client for OpenAI Realtime."""

    def __init__(self, api_key: str, model: str) -> None:
        self.api_key = api_key
        self.model = model
        self._wsapp: websocket.WebSocketApp | None = None
        self._thread: threading.Thread | None = None
        self._opened = threading.Event()
        self._send_queue: "queue.Queue[str | None]" = queue.Queue()

    def start(self) -> None:
        url = f"wss://api.openai.com/v1/realtime?model={self.model}"
        headers = [
            f"Authorization: Bearer {self.api_key}",
            "OpenAI-Beta: realtime=v1",
        ]

        def on_open(wsapp: websocket.WebSocketApp) -> None:
            self._opened.set()

            def sender() -> None:
                while True:
                    item = self._send_queue.get()
                    if item is None:
                        break
                    try:
                        wsapp.send(item)
                    except Exception:
                        break

            threading.Thread(target=sender, daemon=True).start()

        def on_message(_: websocket.WebSocketApp, message: Any) -> None:
            try:
                data = json.loads(message)
            except Exception:
                print("[OpenAI] binary payload", len(message) if isinstance(message, (bytes, bytearray)) else "?")
                return

            event_type = data.get("type")
            if event_type == "response.output_text.delta":
                print(f"[OpenAI] text fragment: {data.get('delta')}")
            elif event_type == "response.completed":
                print("[OpenAI] response completed")
            elif event_type:
                print(f"[OpenAI] {event_type}")

        def on_error(_: websocket.WebSocketApp, error: Any) -> None:
            print(f"[OpenAI] error: {error}")

        def on_close(_: websocket.WebSocketApp, status_code: int, msg: str) -> None:
            print(f"[OpenAI] closed ({status_code}): {msg}")

        self._wsapp = websocket.WebSocketApp(
            url,
            header=headers,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
        )

        def runner() -> None:
            assert self._wsapp is not None
            self._wsapp.run_forever(ping_interval=30, ping_timeout=10)

        self._thread = threading.Thread(target=runner, daemon=True)
        self._thread.start()

    def wait_until_open(self, timeout: float = 10.0) -> bool:
        return self._opened.wait(timeout=timeout)

    def send_json(self, payload: dict[str, Any]) -> None:
        self._send_queue.put(json.dumps(payload))

    def send_audio(self, pcm24: bytes) -> None:
        if not pcm24:
            return
        audio_b64 = base64.b64encode(pcm24).decode()
        self.send_json({"type": "input_audio_buffer.append", "audio": audio_b64})

    def commit_audio(self) -> None:
        self.send_json({"type": "input_audio_buffer.commit"})

    def request_response(self) -> None:
        self.send_json({"type": "response.create", "response": {"modalities": ["text"]}})

    def close(self) -> None:
        self._send_queue.put(None)
        if self._wsapp is not None:
            try:
                self._wsapp.close()
            except Exception:
                pass
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)


@app.websocket("/twilio-stream")
async def twilio_stream(ws: WebSocket) -> None:
    await ws.accept()
    client = OpenAIRealtimeClient(OPENAI_API_KEY, OPENAI_MODEL)
    client.start()

    loop = asyncio.get_running_loop()
    ready = await loop.run_in_executor(None, client.wait_until_open)
    if not ready:
        await ws.close(code=1011)
        client.close()
        return

    rate_state: Any = None

    try:
        while True:
            message = await ws.receive_text()
            try:
                payload = json.loads(message)
            except json.JSONDecodeError:
                print("[Twilio] ignored non-JSON frame")
                continue

            event = payload.get("event")

            if event == "start":
                stream_sid = payload.get("start", {}).get("streamSid")
                print(f"[Twilio] stream started: {stream_sid}")

            elif event == "media":
                media_payload = payload.get("media", {}).get("payload")
                if not media_payload:
                    continue
                mulaw = base64.b64decode(media_payload)
                linear = audioop.ulaw2lin(mulaw, 2)
                pcm24, rate_state = audioop.ratecv(linear, 2, 1, 8000, 24000, rate_state)
                client.send_audio(pcm24)

            elif event == "stop":
                print("[Twilio] stream stopped")
                client.commit_audio()
                client.request_response()
                break

            else:
                print(f"[Twilio] event: {event}")

    except WebSocketDisconnect:
        print("[Twilio] client disconnected")
    finally:
        client.close()
