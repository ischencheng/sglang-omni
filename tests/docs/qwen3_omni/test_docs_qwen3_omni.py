# SPDX-License-Identifier: Apache-2.0
"""Tests for Qwen3-Omni documentation examples.

Every test replicates an API call from `docs/basic_usage/qwen3_omni.md`
so documentation can never silently go stale.

Usage:
    pytest tests/docs/qwen3_omni/test_docs_qwen3_omni.py -s -x
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path

import pytest
import requests

from tests.utils import (
    disable_proxy,
    find_free_port,
    start_server,
    start_server_from_cmd,
    stop_server,
)

MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
MODEL_NAME = "qwen3-omni"

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
IMAGE_PATH = str(DATA_DIR / "cars.jpg")
AUDIO_PATH = str(DATA_DIR / "query_to_cars.wav")
VIDEO_PATH = str(DATA_DIR / "draw.mp4")
VIDEO_AUDIO_PATH = str(DATA_DIR / "query_to_draw.wav")
TEXT_PROMPT = "How many cars are there in the picture?"

STARTUP_TIMEOUT = 900
REQUEST_TIMEOUT = 120


def _post_chat(port: int, payload: dict, timeout: int = REQUEST_TIMEOUT) -> dict:
    """POST to /v1/chat/completions and return the parsed JSON response."""
    with disable_proxy():
        resp = requests.post(
            f"http://localhost:{port}/v1/chat/completions",
            json=payload,
            timeout=timeout,
        )
    resp.raise_for_status()
    return resp.json()


class TestTextOnlyMode:
    """Text-only server (--text-only, single GPU)."""

    @pytest.fixture(scope="class")
    def server(self, tmp_path_factory: pytest.TempPathFactory):
        port = find_free_port()
        log_file = tmp_path_factory.mktemp("text_only_logs") / "server.log"
        proc = start_server(
            MODEL_PATH,
            None,
            log_file,
            port,
            timeout=STARTUP_TIMEOUT,
            extra_args=["--text-only", "--model-name", MODEL_NAME],
        )
        yield port
        stop_server(proc)

    @pytest.mark.docs
    def test_health(self, server: int) -> None:
        """Docs section: Common — Health Check (text-only server)."""
        with disable_proxy():
            resp = requests.get(f"http://localhost:{server}/health", timeout=10)
        assert resp.status_code == 200
        assert "healthy" in resp.text

    @pytest.mark.docs
    def test_image_text(self, server: int) -> None:
        """Docs section: Text-Only Mode — Image and Text Input."""
        result = _post_chat(
            server,
            {
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": TEXT_PROMPT}],
                "images": [IMAGE_PATH],
                "modalities": ["text"],
                "max_tokens": 16,
            },
        )
        assert "choices" in result
        content = result["choices"][0]["message"]["content"]
        assert isinstance(content, str)
        assert len(content) > 0

    @pytest.mark.docs
    def test_audio_image(self, server: int) -> None:
        """Docs section: Text-Only Mode — Audio and Image Input."""
        result = _post_chat(
            server,
            {
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": ""}],
                "images": [IMAGE_PATH],
                "audios": [AUDIO_PATH],
                "modalities": ["text"],
                "max_tokens": 16,
            },
        )
        assert "choices" in result
        content = result["choices"][0]["message"]["content"]
        assert isinstance(content, str)
        assert len(content) > 0


class TestSpeechMode:
    """Speech server (multi-GPU, text + audio output)."""

    @pytest.fixture(scope="class")
    def server(self, tmp_path_factory: pytest.TempPathFactory):
        port = find_free_port()
        log_file = tmp_path_factory.mktemp("speech_logs") / "server.log"
        cmd = [
            sys.executable,
            "examples/run_qwen3_omni_speech_server.py",
            "--model-path",
            MODEL_PATH,
            "--gpu-thinker",
            "0",
            "--gpu-talker",
            "1",
            "--gpu-code-predictor",
            "1",
            "--gpu-code2wav",
            "1",
            "--port",
            str(port),
            "--model-name",
            MODEL_NAME,
        ]
        proc = start_server_from_cmd(cmd, log_file, port, timeout=STARTUP_TIMEOUT)
        yield port
        stop_server(proc)

    @pytest.mark.docs
    def test_health(self, server: int) -> None:
        """Docs section: Common — Health Check (speech server)."""
        with disable_proxy():
            resp = requests.get(f"http://localhost:{server}/health", timeout=10)
        assert resp.status_code == 200
        assert "healthy" in resp.text

    @pytest.mark.docs
    def test_image_text(self, server: int) -> None:
        """Docs section: Speech Mode — Image and Text Input."""
        result = _post_chat(
            server,
            {
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": TEXT_PROMPT}],
                "images": [IMAGE_PATH],
                "modalities": ["text", "audio"],
                "max_tokens": 16,
            },
        )
        assert "choices" in result
        message = result["choices"][0]["message"]

        assert isinstance(message.get("content"), str)
        assert len(message["content"]) > 0

        assert "audio" in message
        audio_b64 = message["audio"]["data"]
        audio_bytes = base64.b64decode(audio_b64)
        assert len(audio_bytes) > 0

    @pytest.mark.docs
    def test_audio_image(self, server: int) -> None:
        """Docs section: Speech Mode — Audio and Image Input."""
        result = _post_chat(
            server,
            {
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": ""}],
                "images": [IMAGE_PATH],
                "audios": [AUDIO_PATH],
                "modalities": ["text", "audio"],
                "max_tokens": 16,
            },
        )
        assert "choices" in result
        message = result["choices"][0]["message"]

        assert isinstance(message.get("content"), str)
        assert len(message["content"]) > 0

        assert "audio" in message
        audio_b64 = message["audio"]["data"]
        audio_bytes = base64.b64decode(audio_b64)
        assert len(audio_bytes) > 0


    @pytest.mark.docs
    def test_video_audio(self, server: int, tmp_path: Path) -> None:
        """Docs section: Speech Mode — Video and Audio Input.

        Verifies:
        1. Text output contains expected keywords about the video content.
        2. Audio output can be transcribed by Whisper ASR and the transcription
           is semantically consistent with the text output.
        """
        result = _post_chat(
            server,
            {
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": ""}],
                "videos": [VIDEO_PATH],
                "audios": [VIDEO_AUDIO_PATH],
                "modalities": ["text", "audio"],
                "max_tokens": 16,
            },
        )
        assert "choices" in result
        message = result["choices"][0]["message"]

        # --- Text output: keyword detection ---
        content = message.get("content", "")
        assert isinstance(content, str)
        assert len(content) > 0
        content_lower = content.lower()
        # draw.mp4 shows a girl drawing with a stylus/pen
        assert any(
            kw in content_lower for kw in ("draw", "stylus", "pen", "tablet", "girl")
        ), f"Text output missing expected keywords about the video. Got: {content}"

        # --- Audio output: Whisper ASR consistency check ---
        assert "audio" in message, "Expected audio in response"
        audio_b64 = message["audio"]["data"]
        audio_bytes = base64.b64decode(audio_b64)
        assert len(audio_bytes) > 0

        wav_path = tmp_path / "video_audio_output.wav"
        wav_path.write_bytes(audio_bytes)

        transcription = _transcribe_with_whisper(str(wav_path))
        assert len(transcription) > 0, "Whisper transcription is empty"

        # Check that the transcription shares at least one significant word
        # with the text output (semantic consistency).
        text_words = set(content_lower.split())
        transcript_words = set(transcription.lower().split())
        # Remove common stop words for a more meaningful overlap check
        stop_words = {"the", "a", "an", "is", "are", "in", "of", "and", "to", "it"}
        text_significant = text_words - stop_words
        transcript_significant = transcript_words - stop_words
        overlap = text_significant & transcript_significant
        assert len(overlap) > 0, (
            f"Talker audio transcription has no semantic overlap with thinker text.\n"
            f"Text output: {content}\n"
            f"Transcription: {transcription}"
        )


def _transcribe_with_whisper(wav_path: str) -> str:
    """Transcribe a WAV file using Whisper ASR."""
    import soundfile as sf
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")

    wav, sr = sf.read(wav_path)
    if sr != 16000:
        import librosa

        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)

    inputs = processor(wav, sampling_rate=16000, return_tensors="pt")
    predicted_ids = model.generate(inputs.input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription.strip()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-s", "-x", "-v"]))
