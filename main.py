import os
import shutil
import tempfile
import subprocess
from typing import List, Dict, Any
import uuid

from enum import Enum
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
import uvicorn
from faster_whisper import WhisperModel


# ===== Enums =====

class ModelSize(str, Enum):
    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE_V2 = "large-v2"
    LARGE_V3 = "large-v3"


class Device(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"


class ComputeType(str, Enum):
    INT8 = "int8"
    FLOAT16 = "float16"
    FLOAT32 = "float32"

app = FastAPI(
    title="Whisper Transcription API",
    version="1.0.0",
)


# ===== Pydantic Models =====

class TranscriptionSegment(BaseModel):
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")
    text: str = Field(..., description="Transcribed text for this segment")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "start": 0.0,
                "end": 2.5,
                "text": "Hello, this is a test."
            }
        }
    )


class TranscriptionResponse(BaseModel):
    language: str = Field(..., description="Detected language code (e.g., 'en')")
    language_probability: float = Field(
        ..., 
        description="Confidence score for language detection (0.0-1.0)", 
        ge=0.0, 
        le=1.0
    )
    segments: List[TranscriptionSegment] = Field(..., description="List of transcribed segments")
    transcription: str = Field(..., description="Full transcribed text")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "language": "en",
                "language_probability": 0.95,
                "segments": [
                    {"start": 0.0, "end": 2.5, "text": "Hello, this is a test."},
                    {"start": 2.5, "end": 5.0, "text": "How are you doing today?"}
                ],
                "transcription": "Hello, this is a test. How are you doing today?"
            }
        }
    )


def extract_audio(input_path: str, output_path: str) -> None:
    """Extract audio from a video or audio file using ffmpeg into WAV 16k mono."""
    command = [
        "ffmpeg",
        "-i",
        input_path,
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        "-y",
        output_path,
    ]
    try:
        subprocess.run(command, check=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg failed: {exc.stderr.decode(errors='ignore')}")


def transcribe_file(
    file_path: str,
    model_size: ModelSize = ModelSize.BASE,
    device: Device = Device.CPU,
    compute_type: ComputeType = ComputeType.INT8,
    work_dir: str | None = None,
) -> TranscriptionResponse:
    """Load model and transcribe the provided audio/video file.

    Returns a TranscriptionResponse with language, language_probability, segments and full transcription.
    """
    # Create temporary WAV path. If a work_dir is provided (the endpoint's tmp dir),
    # place the WAV there so the endpoint can remove the whole directory at once.
    wav_created = False
    if work_dir:
        # Ensure the work_dir exists
        os.makedirs(work_dir, exist_ok=True)
        tmpf = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=work_dir)
        wav_path = tmpf.name
        tmpf.close()
        wav_created = True
    else:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name
            wav_created = True

    try:
        extract_audio(file_path, wav_path)

        model = WhisperModel(model_size.value, device=device.value, compute_type=compute_type.value)
        segments, info = model.transcribe(wav_path, beam_size=5)

        segs: List[TranscriptionSegment] = []
        parts: List[str] = []
        for s in segments:
            segs.append(
                TranscriptionSegment(
                    start=float(s.start),
                    end=float(s.end),
                    text=s.text
                )
            )
            parts.append(s.text)

        full_text = " ".join(parts)

        return TranscriptionResponse(
            language=info.language,
            language_probability=float(info.language_probability),
            segments=segs,
            transcription=full_text,
        )
    finally:
        # Remove the WAV file if we created it. If work_dir was provided, the endpoint
        # typically removes the whole temporary directory after this function returns,
        # but remove the file here as an extra cleanup step to make rmdir succeed.
        try:
            if wav_created and os.path.exists(wav_path):
                os.remove(wav_path)
        except Exception:
            pass


@app.post(
    "/transcribe",
    response_model=TranscriptionResponse,
    responses={
        200: {
            "description": "Successful transcription",
        },
        500: {"description": "Internal server error"},
    },
)
async def transcribe_endpoint(
    file: UploadFile = File(..., description="Audio or video file to transcribe"),
    model_size: ModelSize = Form(ModelSize.BASE, description="Whisper model size"),
    device: Device = Form(Device.CPU, description="Compute device"),
    compute_type: ComputeType = Form(ComputeType.INT8, description="Compute type"),
) -> TranscriptionResponse:
    """Upload a video/audio file and get transcription JSON back.

    **Form fields:**
    - `file`: binary file to transcribe (required)
    - `model_size`: whisper model name (tiny, base, small, medium, large-v2, etc.)
    - `device`: cpu or cuda
    - `compute_type`: int8, float16, or float32

    **Returns:** TranscriptionResponse with transcription details and segments
    """
    # Save upload to a temporary file
    try:
        tmp_dir = tempfile.mkdtemp()
        # Ensure filename is a str (UploadFile.filename can be None) and sanitize it.
        filename: str = "uploaded_file" + uuid.uuid4().hex
        filename = os.path.basename(filename)
        upload_path = os.path.join(tmp_dir, filename)
        with open(upload_path, "wb") as out_file:
            shutil.copyfileobj(file.file, out_file)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {exc}")

    try:
        result = transcribe_file(
            upload_path, model_size=model_size, device=device, compute_type=compute_type, work_dir=tmp_dir
        )
        return result
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        try:
            if os.path.exists(upload_path):
                os.remove(upload_path)
            os.rmdir(tmp_dir)
        except Exception:
            pass


if __name__ == "__main__":
    # Only run this when executed via: python main.py
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)