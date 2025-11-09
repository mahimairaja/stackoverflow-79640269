from transformers import WhisperForConditionalGeneration, WhisperProcessor

_model = None
_processor = None


def load_model():
    global _model, _processor
    if _model is None or _processor is None:
        _processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
        _model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
    return _model, _processor


def get_encoder():
    model, _ = load_model()
    return model.get_encoder()


def get_decoder():
    model, _ = load_model()
    return model.get_decoder()


def get_processor():
    _, processor = load_model()
    return processor
