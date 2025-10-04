#!/bin/bash

# Start script for Twilio-OpenAI Voice Transcription Bridge

echo "🚀 Starting Twilio-OpenAI Voice Transcription Bridge..."
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  Warning: .env file not found!"
    echo "Please create a .env file with your OPENAI_API_KEY"
    echo ""
    echo "Example:"
    echo "OPENAI_API_KEY=sk-your-key-here"
    echo "OPENAI_MODEL=gpt-4o-realtime-preview-2024-12-17"
    echo ""
    exit 1
fi

# Check if OpenAI API key is set
if ! grep -q "OPENAI_API_KEY=sk-" .env; then
    echo "⚠️  Warning: OPENAI_API_KEY not configured in .env file"
    echo "Please add your OpenAI API key to the .env file"
    exit 1
fi

echo "✅ Environment configured"
echo "🌐 Starting server on http://0.0.0.0:8000"
echo ""
echo "📝 To expose this to the internet for Twilio:"
echo "   Run: ngrok http 8000"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Start the server
uvicorn twilio_openai_bridge:app --host 0.0.0.0 --port 8000 --reload

