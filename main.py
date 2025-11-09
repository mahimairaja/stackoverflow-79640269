import asyncio
import json
import logging

import torch
from fastapi import APIRouter, FastAPI, WebSocket

import audio_utils
import inference
import stt_schema

app = FastAPI()
router = APIRouter()


@router.websocket("/stt/predict/live")
async def stt_predict_live(websocket: WebSocket):
    """
    Handles a WebSocket connection for live speech-to-text transcription.

    This function establishes a WebSocket connection with a client, processes
    received audio chunks in real-time, and sends back transcribed text. It
    manages audio buffering and handles any connection or data-related errors
    gracefully. The transcription is performed using defined audio utilities
    and inference models.

    Parameters:
        websocket (WebSocket): The WebSocket connection instance provided by the client.

    Raises:
        asyncio.TimeoutError: Raised when no data is received within the specified timeout.
        ValueError: Raised when invalid or unprocessable audio data is received.
        Exception: Raised for any other unforeseen exceptions during the connection lifecycle.

    Returns:
        None
    """
    await websocket.accept()
    host = websocket.client.host
    port = websocket.client.port

    logging.info(
        "Received WebSocket connection from client host -> %s on port -> %s", host, port
    )
    buffer = torch.empty(0)

    try:
        while True:
            try:
                chunk = await asyncio.wait_for(websocket.receive_bytes(), timeout=5)
            except asyncio.TimeoutError:
                await websocket.send_text(json.dumps({"error": "timeout"}))
                break
            waveform = audio_utils.audio_processing(chunk)
            buffer = torch.cat((buffer, waveform), dim=0)

            if buffer.shape[0] >= audio_utils.BUFFER_SIZE:
                input_features = buffer[-audio_utils.BUFFER_SIZE :]
                transcription = audio_utils.audio_transcription(
                    input_features,
                    inference.get_encoder(),
                    inference.get_decoder(),
                    inference.get_processor(),
                )
                response = stt_schema.TranscriptResponse(transcription=transcription)
                await websocket.send_text(json.dumps(response))
                await asyncio.sleep(0.5)
    except ValueError as invalid_audio:
        await websocket.close(code=1003, reason="Invalid audio")
        logging.error(
            "Invalid audio data received from client host -> %s on port -> %s   error -> %s",
            host,
            port,
            invalid_audio,
        )
    except Exception as e:
        logging.info("Connection closed by remote host or error occurred -> %s", e)
    finally:
        logging.info("Closing WebSocket connection from client host -> %s", host)


app.include_router(router)
