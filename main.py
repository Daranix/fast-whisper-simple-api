import os
import shutil
import tempfile
import subprocess
from typing import List, Dict, Any
import uuid

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from faster_whisper import WhisperModel

app = FastAPI(title="Whisper Transcription API")


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
    model_size: str = "base",
    device: str = "cpu",
    compute_type: str = "int8",
    work_dir: str | None = None,
) -> Dict[str, Any]:
    """Load model and transcribe the provided audio/video file.

    Returns a dict with language, language_probability, segments and full transcription.
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

        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        segments, info = model.transcribe(wav_path, beam_size=5)

        segs = []
        parts: List[str] = []
        for s in segments:
            segs.append({"start": float(s.start), "end": float(s.end), "text": s.text})
            parts.append(s.text)

        full_text = " ".join(parts)

        return {
            "language": info.language,
            "language_probability": float(info.language_probability),
            "segments": segs,
            "transcription": full_text,
        }
    finally:
        # Remove the WAV file if we created it. If work_dir was provided, the endpoint
        # typically removes the whole temporary directory after this function returns,
        # but remove the file here as an extra cleanup step to make rmdir succeed.
        try:
            if wav_created and os.path.exists(wav_path):
                os.remove(wav_path)
        except Exception:
            pass


@app.post("/transcribe")
async def transcribe_endpoint(
    file: UploadFile = File(...),
    model_size: str = Form("base"),
    device: str = Form("cpu"),
    compute_type: str = Form("int8"),
):
    """Upload a video/audio file and get transcription JSON back.

    form fields:
    - file: binary file to transcribe
    - model_size: whisper model name (tiny, base, small, medium, large-v2...)
    - device: cpu or cuda
    - compute_type: int8, float16, float32
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
        return JSONResponse(content=result)
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
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)