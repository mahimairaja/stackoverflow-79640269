from io import BytesIO

import torch
import torchaudio

SAMPLE_RATE = 16000
BUFFER_SIZE = 16000


def audio_processing(audio: bytes):
    """
    Processes an audio input and prepares it for further use by resampling and converting it to a
    single channel if necessary.

    Args:
        audio (bytes): The audio file content in bytes.

    Raises:
        ValueError: If the audio cannot be loaded due to any issue.

    Returns:
        torch.Tensor: A tensor representing the loaded and processed audio waveform.
    """
    try:
        audioFile = BytesIO(audio)
        waveform, sr = torchaudio.load(audioFile)
    except Exception as e:
        raise ValueError(f"failed to load audio : {e}")
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    return waveform


def audio_transcription(waveform, model, processor):
    if waveform.dim() == 2:
        waveform = waveform.squeeze(0)

    waveform_np = (
        waveform.cpu().numpy() if isinstance(waveform, torch.Tensor) else waveform
    )

    input_features = processor(
        waveform_np, sampling_rate=SAMPLE_RATE, return_tensors="pt"
    ).input_features

    device = next(model.parameters()).device
    input_features = input_features.to(device)

    attention_mask = torch.ones(
        input_features.shape[:2], dtype=torch.long, device=device
    )

    with torch.no_grad():
        generated_ids = model.generate(
            input_features,
            attention_mask=attention_mask,
            language="en",
            task="transcribe",
        )

    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return transcription
