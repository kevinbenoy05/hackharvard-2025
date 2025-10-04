"""Twilio media stream bridge with Perplexity AI and OpenAI.

This bridge creates an intelligent phone assistant:
1. Receives audio from Twilio phone calls
2. Transcribes speech using OpenAI Whisper (via Realtime API)
3. Generates detailed responses using Perplexity AI (with web search)
4. Simplifies responses using OpenAI GPT-4o-mini for natural conversation
5. Converts simplified responses to speech using OpenAI TTS
6. Plays back AI voice responses to the caller

Run with: `uvicorn twilio_openai_bridge:app --host 0.0.0.0 --port 8000`
"""

import asyncio
import base64
import json
import os
import queue
import random
import threading
from typing import Any
import math
import struct

import audioop
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import Response
import websocket as ws_client  # websocket-client package
from dotenv import load_dotenv
import httpx
from openai import AsyncOpenAI

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-realtime-preview-2024-12-17")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY before starting the bridge.")
if not PERPLEXITY_API_KEY:
    raise RuntimeError("Set PERPLEXITY_API_KEY before starting the bridge.")

app = FastAPI()

# Initialize OpenAI client for TTS
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)


def generate_thinking_tone(duration_seconds: float = 1.0, frequency: int = 440) -> bytes:
    """Generate a simple pleasant tone for 'thinking' indicator.
    
    Args:
        duration_seconds: Duration of the tone
        frequency: Frequency in Hz (440 = A note, pleasant)
    
    Returns:
        bytes: PCM16 audio data at 8kHz (ready for μ-law conversion)
    """
    sample_rate = 8000  # Twilio expects 8kHz
    num_samples = int(sample_rate * duration_seconds)
    
    # Generate a soft sine wave
    audio_data = bytearray()
    for i in range(num_samples):
        # Sine wave with fade in/out for smoother sound
        t = i / sample_rate
        fade = 1.0
        if t < 0.05:  # Fade in first 50ms
            fade = t / 0.05
        elif t > duration_seconds - 0.05:  # Fade out last 50ms
            fade = (duration_seconds - t) / 0.05
        
        # Generate soft tone (amplitude 0.3 for gentle sound)
        sample = int(32767 * 0.3 * fade * math.sin(2 * math.pi * frequency * t))
        # Convert to 16-bit PCM
        audio_data.extend(struct.pack('<h', sample))
    
    return bytes(audio_data)


async def play_thinking_sound_loop(callback, stream_sid: str, stop_event: asyncio.Event):
    """Play thinking sound starting from a random position until stop_event is set.
    
    This plays the audio while AI processes the response, starting from a random
    position in the file to add variety, then stops when the response is ready.
    
    Args:
        callback: Function to send audio chunks to Twilio
        stream_sid: Twilio stream ID
        stop_event: Event that signals when to stop playing
    """
    try:
        # Load and prepare audio data once
        thinking_wav_path = "autumn.wav"
        if os.path.exists(thinking_wav_path):
            import wave
            with wave.open(thinking_wav_path, 'rb') as wav:
                # Get WAV file properties
                channels = wav.getnchannels()
                sample_width = wav.getsampwidth()
                framerate = wav.getframerate()
                
                print(f"[DEBUG] WAV file: {channels}ch, {framerate}Hz, {sample_width*8}bit")
                
                # Read all audio data
                pcm_data = wav.readframes(wav.getnframes())
                
                # Convert stereo to mono if needed
                if channels == 2:
                    pcm_data = audioop.tomono(pcm_data, sample_width, 1, 1)
                    print(f"[DEBUG] Converted stereo to mono")
                
                # Convert to 16-bit if not already
                if sample_width != 2:
                    if sample_width == 1:  # 8-bit
                        pcm_data = audioop.lin2lin(pcm_data, 1, 2)
                    elif sample_width == 4:  # 32-bit
                        pcm_data = audioop.lin2lin(pcm_data, 4, 2)
                    print(f"[DEBUG] Converted {sample_width*8}bit to 16bit")
                
                # Resample to 8kHz for Twilio (if not already)
                if framerate != 8000:
                    pcm_data, _ = audioop.ratecv(pcm_data, 2, 1, framerate, 8000, None)
                    print(f"[DEBUG] Resampled {framerate}Hz to 8000Hz")
                
                # Reduce volume to 30% to prevent it being too loud
                pcm_data = audioop.mul(pcm_data, 2, 0.3)
                print(f"[DEBUG] Reduced volume to 30%")
        else:
            print(f"[DEBUG] WAV file not found, using generated tone")
            # Fallback to generated tone
            pcm_data = generate_thinking_tone(duration_seconds=3.0, frequency=440)
        
        # Convert to μ-law for Twilio once
        mulaw_data = audioop.lin2ulaw(pcm_data, 2)
        chunk_size = 160  # 20ms chunks at 8kHz
        
        # Pick a random starting position (aligned to chunk boundaries)
        total_chunks = len(mulaw_data) // chunk_size
        if total_chunks > 100:  # Only randomize if file is long enough (>2 seconds)
            # Start somewhere in the first 80% of the file
            max_start_chunk = int(total_chunks * 0.8)
            start_chunk = random.randint(0, max_start_chunk)
            start_byte = start_chunk * chunk_size
            print(f"[DEBUG] Starting playback from {start_byte}/{len(mulaw_data)} bytes (chunk {start_chunk}/{total_chunks})")
        else:
            start_byte = 0
            print(f"[DEBUG] Playing from beginning")
        
        # Play from the random start position until stopped
        bytes_played = 0
        for i in range(start_byte, len(mulaw_data), chunk_size):
            if stop_event.is_set():
                print(f"[DEBUG] Thinking sound stopped after {bytes_played} bytes")
                return
            
            chunk = mulaw_data[i:i + chunk_size]
            callback(chunk, stream_sid)
            bytes_played += len(chunk)
            await asyncio.sleep(0.02)  # 20ms between chunks
        
        print(f"[DEBUG] Thinking sound completed ({bytes_played} bytes played)")
    except asyncio.CancelledError:
        print(f"[DEBUG] Thinking sound cancelled")
        raise
    except Exception as e:
        print(f"[ERROR] Failed to play thinking sound: {e}")


async def play_ready_beep(callback, stream_sid: str):
    """Play a short beep to signal the system is ready to listen again.
    
    Args:
        callback: Function to send audio chunks to Twilio
        stream_sid: Twilio stream ID
    """
    try:
        # Generate a pleasant two-tone beep (like a notification sound)
        # First tone: higher pitch, second tone: lower pitch
        sample_rate = 8000
        
        # First beep: 800Hz for 0.1s
        beep1 = generate_thinking_tone(duration_seconds=0.1, frequency=800)
        # Brief silence: 0.05s
        silence_samples = int(sample_rate * 0.05)
        silence = bytes(silence_samples * 2)  # 16-bit silence
        # Second beep: 600Hz for 0.1s
        beep2 = generate_thinking_tone(duration_seconds=0.1, frequency=600)
        
        # Combine the beeps
        pcm_data = beep1 + silence + beep2
        
        # Convert to μ-law for Twilio
        mulaw_data = audioop.lin2ulaw(pcm_data, 2)
        
        # Send in chunks
        chunk_size = 160
        for i in range(0, len(mulaw_data), chunk_size):
            chunk = mulaw_data[i:i + chunk_size]
            callback(chunk, stream_sid)
            await asyncio.sleep(0.02)
        
        print(f"[DEBUG] Ready beep played")
    except Exception as e:
        print(f"[ERROR] Failed to play ready beep: {e}")


async def get_perplexity_response(messages: list[dict]) -> str:
    """Get a response from Perplexity API."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "https://api.perplexity.ai/chat/completions",
                headers={
                    "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "sonar",  # Fast, lightweight search model
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 300,  # Keep responses concise for phone calls
                    "search_mode": "web"
                },
                timeout=30.0
            )
            
            # Log the error response for debugging
            if response.status_code != 200:
                error_text = response.text
                print(f"[DEBUG] Perplexity API error: {response.status_code} - {error_text}")
                
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            print(f"[ERROR] Perplexity API error: {e.response.status_code}")
            print(f"[ERROR] Response: {e.response.text}")
            # Return a fallback response
            return "I apologize, but I'm having trouble accessing my knowledge base right now. Could you try asking me again in a moment?"


async def simplify_response(text: str) -> str:
    """Use OpenAI to simplify and shorten a response for phone conversation."""
    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are helping convert detailed text responses into concise, natural phone conversation responses. Keep it under 50 words, conversational, and easy to understand when spoken aloud. Remove citations and keep only the most important information."
                },
                {
                    "role": "user",
                    "content": f"Simplify this response for a phone call:\n\n{text}"
                }
            ],
            temperature=0.3,
            max_tokens=150
        )
        simplified = response.choices[0].message.content
        return simplified if simplified else text
    except Exception as e:
        print(f"[ERROR] Simplification failed: {e}")
        # Fallback: just truncate to first 2 sentences
        sentences = text.split('.')[:2]
        return '. '.join(sentences) + '.'


async def text_to_speech_stream(text: str, callback, stream_sid: str) -> float:
    """Convert text to speech using OpenAI TTS and stream to Twilio.
    
    Returns:
        float: Duration of the audio in seconds
    """
    try:
        # Generate speech using OpenAI TTS
        response = await openai_client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text,
            response_format="pcm"  # 24kHz PCM16 mono
        )
        
        # Read the response content and process in chunks
        audio_data = response.content
        total_bytes = len(audio_data)
        
        print(f"[DEBUG] TTS generated {total_bytes} bytes at 24kHz PCM16")
        
        # Process and stream audio in chunks
        # Use smaller chunks for smoother playback
        chunk_size = 3840  # Multiple of 160 for better alignment (160 * 24)
        rate_state = None
        
        for i in range(0, total_bytes, chunk_size):
            chunk = audio_data[i:i + chunk_size]
            if chunk:
                # Ensure chunk has even number of bytes (for 16-bit samples)
                if len(chunk) % 2 != 0:
                    chunk = chunk[:-1]
                
                # Convert PCM 24kHz to 8kHz for Twilio
                pcm8, rate_state = audioop.ratecv(chunk, 2, 1, 24000, 8000, rate_state)
                # Convert PCM to μ-law
                mulaw = audioop.lin2ulaw(pcm8, 2)
                
                callback(mulaw, stream_sid)
                
                # Small delay to prevent overwhelming the connection
                # 20ms chunks for smooth playback
                await asyncio.sleep(0.02)
        
        # Calculate audio duration: PCM16 at 24kHz, mono
        # Duration = total_samples / sample_rate
        # total_samples = total_bytes / bytes_per_sample
        # For PCM16: bytes_per_sample = 2
        duration_seconds = total_bytes / (24000 * 2)
        print(f"[DEBUG] TTS duration: {duration_seconds:.2f}s")
        return duration_seconds
    except Exception as e:
        print(f"[ERROR] TTS failed: {e}")
        return 0.0


@app.get("/")
async def root():
    return {"status": "Twilio-OpenAI Bridge is running"}


@app.post("/twilio-voice")
async def twilio_voice(request: Request):
    """Twilio webhook for incoming calls. Returns TwiML to start media stream."""
    # Get the host from the request to construct the WebSocket URL
    host = request.headers.get("host", "localhost:8000")
    protocol = "wss" if request.url.scheme == "https" else "ws"
    ws_url = f"{protocol}://{host}/twilio-stream"
    
    twiml = f'''<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="{ws_url}" />
    </Connect>
</Response>'''
    
    return Response(content=twiml, media_type="application/xml")


class OpenAIRealtimeClient:
    """OpenAI Realtime client for transcription only."""

    def __init__(self, api_key: str, model: str, transcription_callback=None) -> None:
        self.api_key = api_key
        self.model = model
        self.transcription_callback = transcription_callback  # Callback when transcription completes
        self._wsapp: ws_client.WebSocketApp | None = None
        self._thread: threading.Thread | None = None
        self._opened = threading.Event()
        self._send_queue: "queue.Queue[str | None]" = queue.Queue()

    def start(self) -> None:
        url = f"wss://api.openai.com/v1/realtime?model={self.model}"
        headers = [
            f"Authorization: Bearer {self.api_key}",
            "OpenAI-Beta: realtime=v1",
        ]

        def on_open(wsapp: ws_client.WebSocketApp) -> None:
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

        def on_message(_: ws_client.WebSocketApp, message: Any) -> None:
            try:
                data = json.loads(message)
            except Exception:
                return

            event_type = data.get("type")
            
            # Handle transcription events
            if event_type == "conversation.item.input_audio_transcription.completed":
                transcript = data.get("transcript", "")
                print(f"[USER] {transcript}")
                # Trigger callback with transcription
                if self.transcription_callback and transcript.strip():
                    self.transcription_callback(transcript)
            elif event_type == "conversation.item.input_audio_transcription.failed":
                error = data.get("error", {})
                print(f"[ERROR] Transcription failed: {error}")

        def on_error(_: ws_client.WebSocketApp, error: Any) -> None:
            print(f"[ERROR] OpenAI connection error: {error}")

        def on_close(_: ws_client.WebSocketApp, status_code: int, msg: str) -> None:
            pass  # Silent close

        self._wsapp = ws_client.WebSocketApp(
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

    def configure_session(self) -> None:
        """Configure session for transcription only."""
        self.send_json({
            "type": "session.update",
            "session": {
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,  # Voice detection sensitivity (0.0-1.0)
                    "prefix_padding_ms": 300,  # Audio before speech to include
                    "silence_duration_ms": 1500  # Wait 1.5s of silence before ending turn
                },
                "input_audio_format": "pcm16",
                "input_audio_transcription": {
                    "model": "whisper-1"
                }
            }
        })

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
    
    loop = asyncio.get_running_loop()
    stream_sid: str | None = None
    conversation_history: list[dict] = [
        {"role": "system", "content": "You are a helpful AI assistant on a phone call. Provide accurate, informative responses. They will be simplified for voice, so you can include detailed information."}
    ]
    # Flag to prevent processing new transcriptions while AI is thinking/speaking
    # Audio continues to be transcribed, but transcriptions are ignored until:
    # 1. Perplexity generates response
    # 2. OpenAI simplifies the response
    # 3. TTS audio is fully played back to the caller
    is_processing = False
    
    # Callback to send audio back to Twilio
    def send_audio_to_twilio(mulaw_audio: bytes, sid: str):
        """Send audio back to Twilio (called from TTS thread)."""
        try:
            audio_b64 = base64.b64encode(mulaw_audio).decode()
            media_message = {
                "event": "media",
                "streamSid": sid,
                "media": {
                    "payload": audio_b64
                }
            }
            # Schedule the send on the event loop
            asyncio.run_coroutine_threadsafe(
                ws.send_text(json.dumps(media_message)),
                loop
            )
        except Exception as e:
            print(f"[ERROR] Failed to send audio to Twilio: {e}")
    
    # Callback when transcription is received
    def on_transcription(transcript: str):
        """Handle incoming transcription from user."""
        nonlocal is_processing
        if is_processing:
            print(f"[IGNORED] '{transcript}' (AI is speaking)")
            return  # Skip if already processing - AI is thinking or speaking
        
        is_processing = True
        
        async def process_and_respond():
            nonlocal is_processing
            try:
                # Add user message to history
                conversation_history.append({"role": "user", "content": transcript})
                
                # Start playing thinking sound in a loop (plays until we stop it)
                stop_thinking = asyncio.Event()
                thinking_task = None
                if stream_sid:
                    print("🔔 Starting thinking sound loop...")
                    thinking_task = asyncio.create_task(
                        play_thinking_sound_loop(send_audio_to_twilio, stream_sid, stop_thinking)
                    )
                
                # Get response from Perplexity (while music plays in background)
                print("🤖 Getting Perplexity response...")
                perplexity_response = await get_perplexity_response(conversation_history)
                print(f"[PERPLEXITY] {perplexity_response[:200]}...")  # Show preview
                
                # Simplify the response using OpenAI (music still playing)
                print("✨ Simplifying response...")
                simplified_response = await simplify_response(perplexity_response)
                print(f"[SIMPLIFIED] {simplified_response}")
                
                # Add simplified response to history
                conversation_history.append({"role": "assistant", "content": simplified_response})
                
                # NOW stop the thinking sound before playing TTS
                if thinking_task:
                    print("🛑 Stopping thinking sound...")
                    stop_thinking.set()
                    try:
                        await asyncio.wait_for(thinking_task, timeout=1.0)
                    except asyncio.TimeoutError:
                        thinking_task.cancel()
                    # Small pause to ensure clean transition
                    await asyncio.sleep(0.1)
                
                # Convert to speech and stream back
                if stream_sid:
                    print("🔊 Playing response...")
                    audio_duration = await text_to_speech_stream(simplified_response, send_audio_to_twilio, stream_sid)
                    
                    # Wait for the audio to finish playing before accepting new input
                    # Add a small buffer to account for network delays
                    print(f"⏳ Audio duration: {audio_duration:.1f}s - waiting for playback to complete...")
                    await asyncio.sleep(audio_duration)  # +0.5s buffer
                    
            except Exception as e:
                print(f"[ERROR] Failed to process response: {e}")
            finally:
                is_processing = False
        
        # Schedule the async processing
        asyncio.run_coroutine_threadsafe(process_and_respond(), loop)
    
    client = OpenAIRealtimeClient(OPENAI_API_KEY, OPENAI_MODEL, transcription_callback=on_transcription)
    client.start()

    ready = await loop.run_in_executor(None, client.wait_until_open)
    if not ready:
        print("[ERROR] Failed to connect to OpenAI")
        await ws.close(code=1011)
        client.close()
        return

    # Enable transcription
    client.configure_session()
    print("📞 Call connected - listening...")

    rate_state: Any = None

    try:
        while True:
            message = await ws.receive_text()
            try:
                payload = json.loads(message)
            except json.JSONDecodeError:
                continue

            event = payload.get("event")

            if event == "start":
                # Get the stream SID for audio playback
                stream_sid = payload.get("start", {}).get("streamSid")

            elif event == "media":
                media_payload = payload.get("media", {}).get("payload")
                if not media_payload:
                    continue
                # Convert Twilio's μ-law audio to PCM16 24kHz for OpenAI
                mulaw = base64.b64decode(media_payload)
                linear = audioop.ulaw2lin(mulaw, 2)
                pcm24, rate_state = audioop.ratecv(linear, 2, 1, 8000, 24000, rate_state)
                client.send_audio(pcm24)

            elif event == "stop":
                print("📞 Call ended\n")
                # Commit any remaining audio for transcription
                client.commit_audio()
                break

    except WebSocketDisconnect:
        print("[ERROR] Client disconnected unexpectedly")
    finally:
        client.close()
