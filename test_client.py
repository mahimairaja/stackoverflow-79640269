import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

import websockets

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def send_audio_chunks(uri: str, audio_file: Path, chunk_size: int = 4096):
    try:
        async with websockets.connect(uri) as websocket:
            logger.info(f"Connected to {uri}")
            logger.info(f"Sending audio file: {audio_file}")

            with open(audio_file, "rb") as f:
                chunk_count = 0
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break

                    await websocket.send(chunk)
                    chunk_count += 1
                    logger.debug(f"Sent chunk {chunk_count} ({len(chunk)} bytes)")

                    await asyncio.sleep(0.1)

            logger.info(f"Finished sending {chunk_count} chunks")

            try:
                while True:
                    response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                    data = json.loads(response)

                    if "error" in data:
                        logger.error(f"Received error: {data['error']}")
                        break
                    elif "transcription" in data:
                        logger.info(f"Transcription: {data['transcription']}")
                    else:
                        logger.info(f"Received: {data}")

            except asyncio.TimeoutError:
                logger.info("No more responses, closing connection")

    except websockets.exceptions.ConnectionClosed:
        logger.info("Connection closed by server")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)


async def main():
    parser = argparse.ArgumentParser(description="Test WebSocket STT service")
    parser.add_argument(
        "--host", default="localhost", help="Server host (default: localhost)"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Server port (default: 8000)"
    )
    parser.add_argument(
        "--audio-file",
        type=Path,
        help="Path to audio file to send (WAV format recommended)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=4096,
        help="Size of each audio chunk in bytes (default: 4096)",
    )

    args = parser.parse_args()

    uri = f"ws://{args.host}:{args.port}/stt/predict/live"

    audio_file = args.audio_file

    if not audio_file.exists():
        logger.error(f"Audio file not found: {audio_file}")
        sys.exit(1)

    await send_audio_chunks(uri, audio_file, args.chunk_size)


if __name__ == "__main__":
    asyncio.run(main())
