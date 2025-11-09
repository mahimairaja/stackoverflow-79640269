# StackOverflow fix for 79640269 

A real-time speech-to-text service using OpenAI's Whisper model via WebSocket connections. This is a fix for the issue described in [StackOverflow 79640269](https://stackoverflow.com/questions/79640269/websocket-stt-service-not-working-as-expected).


## Installation

Install dependencies using `uv`:

```bash
uv sync
```

## Running the Server

Start the FastAPI server:

```bash
$ uvicorn main:app --host 0.0.0.0 --port 8000
```

The server will:
- Download the Whisper-tiny model on first run (if not cached)
- Start listening on `http://localhost:8000`
- WebSocket endpoint available at `ws://localhost:8000/stt/predict/live`

## Testing

### Using the Test Client

A test client script is provided to verify the service:

```bash
$ python test_client.py --audio-file sample.wav
```

### Test Client Requirements

The test client requires additional dependencies:

```bash
pip install websockets soundfile numpy
```
