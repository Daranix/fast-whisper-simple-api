# Fast Whisper Simple API

A lightweight, production-ready FastAPI service for speech-to-text transcription using [faster-whisper](https://github.com/SYSTRAN/faster-whisper). Upload audio or video files and get JSON transcriptions with language detection, timing information, and full text.

## Features

- 🚀 **FastAPI** — modern async web framework
- 🎯 **faster-whisper** — fast, accurate speech recognition
- 🐳 **Multiple Docker images** — slim (CPU), Alpine, and NVIDIA GPU variants
- 📦 **uv-based builds** — reproducible, fast multistage Docker builds
- 🎵 **Audio/Video support** — handles MP4, WAV, MP3, and more via ffmpeg
- 🏷️ **Language detection** — automatic language identification with confidence scores
- ⏱️ **Segment timestamps** — precise start/end times for each transcribed segment
- 🗑️ **Temp cleanup** — automatic cleanup of uploaded files and extracted audio

## Quick Start

### Local Setup

Requires Python 3.13+ and [uv](https://github.com/astral-sh/uv).

1. **Install dependencies:**
   ```bash
   uv sync
   ```

2. **Run the API:**
   ```bash
   python main.py
   ```
   Or with uvicorn directly for custom settings:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
   ```

3. **Access the API:**
   - Web UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

### Docker

Two image variants are available at `ghcr.io/daranix/fast-whisper-simple-api`:

#### Slim (CPU - Debian bookworm)
```bash
docker pull ghcr.io/daranix/fast-whisper-simple-api:latest

docker run -p 8000:8000 \
  ghcr.io/daranix/fast-whisper-simple-api:latest
```

#### GPU (NVIDIA CUDA 12.2)
```bash
docker pull ghcr.io/daranix/fast-whisper-simple-api:gpu

docker run --gpus all -p 8000:8000 \
  ghcr.io/daranix/fast-whisper-simple-api:gpu
```

**Note:** Alpine is not supported. The project dependencies (`faster-whisper`, `torch`, `onnxruntime`) do not provide pre-built wheels for musl/Alpine. The slim Debian image is already lightweight (~200MB) and is the recommended CPU variant.

## API Usage

### Transcribe Endpoint

**POST** `/transcribe`

Upload a media file and receive a JSON transcription.

#### Form Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | binary | **required** | Audio or video file (MP3, WAV, MP4, etc.) |
| `model_size` | string | `base` | Whisper model size: `tiny`, `base`, `small`, `medium`, `large-v2`, `large-v3` |
| `device` | string | `cpu` | Device: `cpu` or `cuda` (GPU requires NVIDIA image) |
| `compute_type` | string | `int8` | Precision: `int8`, `float16`, `float32` |

#### Response

```json
{
  "language": "en",
  "language_probability": 0.95,
  "segments": [
    {
      "start": 0.0,
      "end": 2.5,
      "text": "Hello, this is a test."
    },
    {
      "start": 2.5,
      "end": 5.0,
      "text": "How are you doing today?"
    }
  ],
  "transcription": "Hello, this is a test. How are you doing today?"
}
```

#### Examples

**Using curl:**

```bash
# Simple transcription
curl -X POST "http://localhost:8000/transcribe" \
  -F "file=@audio.mp3"

# With custom model and device
curl -X POST "http://localhost:8000/transcribe" \
  -F "file=@video.mp4" \
  -F "model_size=small" \
  -F "device=cpu" \
  -F "compute_type=float16"
```

**Using Python:**

```python
import requests

with open("audio.mp3", "rb") as f:
    response = requests.post(
        "http://localhost:8000/transcribe",
        files={"file": f},
        data={
            "model_size": "base",
            "device": "cpu",
            "compute_type": "int8"
        }
    )
    print(response.json())
```

**Using JavaScript/Node.js:**

```javascript
const formData = new FormData();
formData.append("file", audioFile);
formData.append("model_size", "base");
formData.append("device", "cpu");
formData.append("compute_type", "int8");

const response = await fetch("http://localhost:8000/transcribe", {
  method: "POST",
  body: formData
});

const result = await response.json();
console.log(result);
```

## Building Docker Images Locally

### Build all images:

```bash
# Slim (CPU) - Debian based
docker build -t whisper-api:latest .

# GPU - NVIDIA CUDA
docker build -f Dockerfile.nvidia -t whisper-api:gpu .
```

### Build with Docker Buildx (recommended):

```bash
# Enable buildx
docker buildx create --use

# Build and load
docker buildx build -t whisper-api:latest --load .
docker buildx build -f Dockerfile.nvidia -t whisper-api:gpu --load .
```

## Architecture

### Multistage Builds with uv

All Dockerfiles use a multistage build pattern:

1. **Builder stage** (`ghcr.io/astral-sh/uv:*`): 
   - Installs managed Python 3.13 via uv
   - Caches dependencies via `uv sync`
   - Compiles bytecode for faster startup

2. **Final stage**:
   - Lightweight runtime image (Debian slim, Alpine, or NVIDIA CUDA)
   - Copies Python installation and app from builder
   - Non-root user for security
   - ~200MB for slim, ~150MB for Alpine, ~3GB for GPU

### Why uv?

- **Fast**: 5-10x faster than pip
- **Reliable**: Reproducible builds with locked dependencies
- **Portable**: Bundled Python via managed installs
- **Cache-friendly**: BuildKit layer caching for faster rebuilds

## Model Sizes and Performance

| Model | Size | Speed | Accuracy | Memory |
|-------|------|-------|----------|--------|
| `tiny` | 39M | Very Fast | Fair | ~1GB |
| `base` | 140M | Fast | Good | ~2GB |
| `small` | 244M | Good | Very Good | ~3GB |
| `medium` | 769M | Slower | Excellent | ~5GB |
| `large-v2` | 1.5B | Slow | Best | ~10GB |
| `large-v3` | 1.5B | Slow | Best (v3) | ~10GB |

## Compute Type Trade-offs

| Type | Speed | Accuracy | Memory | Use Case |
|------|-------|----------|--------|----------|
| `int8` | Fastest | Good | Low | Default, balanced |
| `float16` | Fast | Better | Medium | GPU preferred |
| `float32` | Slower | Best | High | CPU, precision required |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8000` | Server port |
| `WORKERS` | `4` | Number of worker processes for parallel request handling |

Example:
```bash
docker run -p 9000:9000 -e PORT=9000 -e WORKERS=2 ghcr.io/daranix/fast-whisper-simple-api:latest
```

## File Handling

- **Uploaded files** are stored in a temporary directory
- **Extracted audio** is created as WAV format (16kHz, mono, PCM)
- **All files are deleted** after transcription completes (even on error)
- Max file size depends on available disk space

## Troubleshooting

### Out of Memory
Use a smaller model or increase container memory:
```bash
docker run -m 4g -p 8000:8000 ghcr.io/daranix/fast-whisper-simple-api:latest
```

### Slow Transcription
- Use `int8` compute type (default, faster)
- Use smaller model (`tiny`, `base`, or `small`)
- For GPU, ensure NVIDIA drivers are installed and accessible

### GPU Not Detected
```bash
# Check if GPU is available in container
docker run --gpus all ghcr.io/daranix/fast-whisper-simple-api:gpu \
  python -c "import torch; print(torch.cuda.is_available())"
```

### Audio Format Issues
Ensure ffmpeg is available (installed in all Docker images). Test locally:
```bash
ffmpeg -i yourfile.mp4 -vn -acodec pcm_s16le -ar 16000 -ac 1 test.wav
```

## Development

### Install dev dependencies:
```bash
uv sync  # includes dev dependencies if configured
```

### Run with auto-reload:
```bash
fastapi dev main:app
```

### Run tests (if added):
```bash
uv run pytest
```

## Deployment

### Docker Compose Example:

```yaml
version: '3.8'

services:
  whisper-api:
    image: ghcr.io/daranix/fast-whisper-simple-api:latest
    ports:
      - "8000:8000"
    environment:
      - PORT=8000
    volumes:
      - ./uploads:/tmp  # Optional: persist uploads
    restart: unless-stopped
```

Run with:
```bash
docker-compose up -d
```

### Kubernetes Example:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: whisper-api
spec:
  containers:
  - name: api
    image: ghcr.io/daranix/fast-whisper-simple-api:latest
    ports:
    - containerPort: 8000
    resources:
      requests:
        memory: "2Gi"
        cpu: "1"
      limits:
        memory: "4Gi"
        cpu: "2"
```

## Performance Tips

1. **Pre-warm the model** by making a test request after deployment
2. **Use persistent volume** for model cache (mounted at `~/.cache/whisper`)
3. **Batch requests** if possible (process multiple files in sequence)
4. **Monitor memory** during deployment; models can use 2-10GB
5. **Use GPU** for medium/large models (20-100x faster)

## CI/CD

### Automated Image Builds

Images are built and pushed to `ghcr.io/daranix/fast-whisper-simple-api` on release:

- Create a GitHub release tag (e.g., `v1.0.0`)
- GitHub Actions automatically:
  - Builds two image variants (slim CPU and GPU)
  - Tags with `:latest`, `:slim` (CPU)
  - Tags with `:gpu`
  - Also tags with version (e.g., `:slim-v1.0.0`, `:gpu-v1.0.0`)
  - Pushes to GitHub Container Registry

Workflow: `.github/workflows/deploy.yml`

## License

MIT

## Contributing

Pull requests welcome! Please ensure:
- Code follows project style
- Changes are tested (if applicable)
- README is updated for new features

## Support

For issues, feature requests, or questions, please open an issue on GitHub:
https://github.com/Daranix/fast-whisper-simple-api/issues
