# Twilio Voice Transcription Setup Guide

This guide will help you set up a system where you can call a Twilio number, speak into the phone, and have your speech transcribed in real-time using OpenAI's Realtime API.

## Prerequisites

1. **Twilio Account**: Sign up at [twilio.com](https://www.twilio.com)
2. **OpenAI API Key**: Get it from [platform.openai.com](https://platform.openai.com)
3. **ngrok or similar tool**: For exposing your local server to the internet

## Setup Steps

### 1. Install Dependencies

```bash
# Install Python dependencies using uv
uv sync
```

### 2. Configure Environment Variables

Create a `.env` file in the backend directory:

```bash
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:
```
OPENAI_API_KEY=sk-...your-key-here...
OPENAI_MODEL=gpt-4o-realtime-preview-2024-12-17
```

### 3. Start the Server

```bash
uvicorn twilio_openai_bridge:app --host 0.0.0.0 --port 8000
```

You should see:
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 4. Expose Your Server to the Internet

In a new terminal, use ngrok to create a public URL:

```bash
ngrok http 8000
```

You'll see output like:
```
Forwarding   https://abc123.ngrok.io -> http://localhost:8000
```

Copy the `https://` URL (e.g., `https://abc123.ngrok.io`).

### 5. Configure Your Twilio Phone Number

1. Go to [Twilio Console](https://console.twilio.com/us1/develop/phone-numbers/manage/incoming)
2. Click on your phone number
3. Scroll to "Voice Configuration"
4. Under "A CALL COMES IN", set:
   - **Webhook**: `https://your-ngrok-url.ngrok.io/twilio-voice`
   - **HTTP Method**: POST
5. Click "Save"

## Testing

1. **Call your Twilio number** from any phone
2. You'll hear: "Please speak after the beep, and your speech will be transcribed."
3. **Speak into the phone**
4. Watch your terminal for transcriptions!

### Expected Output

```
[Twilio] WebSocket connection accepted
[OpenAI] Session created
[System] Transcription enabled - ready to receive audio
[Twilio] stream started
  Stream SID: MZ...
  Call SID: CA...

============================================================
  CALL CONNECTED - Speak into the phone
============================================================

[OpenAI] Speech detected, listening...
[OpenAI] Speech stopped
[OpenAI] Audio committed, processing transcription...

[TRANSCRIPTION] Hello, this is a test of the transcription system.

[Twilio] stream stopped - call ended
[System] Connection closed
```

## Architecture

```
┌─────────┐      ┌─────────┐      ┌──────────┐      ┌────────┐
│  Phone  │─────>│ Twilio  │─────>│  FastAPI │─────>│ OpenAI │
│  Call   │      │ Stream  │      │ WebSocket│      │ Realtime│
└─────────┘      └─────────┘      └──────────┘      └────────┘
                   (μ-law)          (PCM24)         (Transcribe)
```

## How It Works

1. **Incoming Call**: When someone calls your Twilio number, Twilio sends a webhook to `/twilio-voice`
2. **TwiML Response**: The server returns TwiML that tells Twilio to start a media stream to `/twilio-stream`
3. **Audio Streaming**: Twilio streams μ-law encoded audio chunks via WebSocket
4. **Audio Conversion**: The server converts μ-law to PCM24 format (required by OpenAI)
5. **OpenAI Processing**: Audio is sent to OpenAI Realtime API with transcription enabled
6. **Voice Activity Detection**: OpenAI detects when you start/stop speaking
7. **Transcription**: When speech stops, OpenAI transcribes it using Whisper
8. **Output**: Transcriptions are printed to the terminal

## Troubleshooting

### No transcriptions appearing?

- Check that your OpenAI API key is valid
- Ensure you're speaking clearly and loudly enough
- Wait a moment after speaking (transcription happens after speech stops)

### Twilio not connecting?

- Verify your ngrok URL is correct in Twilio console
- Make sure the webhook URL ends with `/twilio-voice`
- Check that your server is running

### WebSocket errors?

- Ensure the ngrok URL uses `https://` (not `http://`)
- The system auto-detects protocol but ngrok should provide HTTPS

## Cost Considerations

- **Twilio**: Pay per minute of phone calls
- **OpenAI Realtime API**: Pay per audio minute and token usage
- Both services charge based on usage

## Next Steps

- Modify the transcription handling in `twilio_openai_bridge.py` to save transcriptions to a database
- Add speaker identification
- Implement real-time translation
- Build a web dashboard to view transcriptions

## API Endpoints

- `GET /` - Health check
- `POST /twilio-voice` - Twilio webhook (returns TwiML)
- `WebSocket /twilio-stream` - Media stream endpoint

