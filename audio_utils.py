from io import BytesIO

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


def audio_transcription(input_features, encoder, decoder, processor):
    """
    Transcribes audio input using a pre-trained model.

    Args:
        input_features (torch.Tensor): The input features to transcribe.
        encoder (torch.nn.Module): The encoder model.
        decoder (torch.nn.Module): The decoder model.
        processor (torch.nn.Module): The processor model.
    """
    return encoder(input_features).logits.argmax(dim=-1)
