## Builder: use uv image to provision a managed Python and install dependencies
FROM ghcr.io/astral-sh/uv:bookworm-slim AS builder
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
ENV UV_PYTHON_INSTALL_DIR=/python
ENV UV_PYTHON_PREFERENCE=only-managed

# Install a managed Python version for the builder
RUN uv python install 3.13

WORKDIR /app

# Use build cache mounts to speed up dependency installs. Bind pyproject.toml so
# buildkit can cache dependency resolution separately from source changes.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --no-install-project --no-dev || true

COPY . /app

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev

## Final image: lightweight Debian bookworm-slim
FROM debian:bookworm-slim

# Setup a non-root user
RUN groupadd --system --gid 999 nonroot \
 && useradd --system --gid 999 --uid 999 --create-home nonroot

# Install runtime system deps (ffmpeg for audio extraction, libsndfile)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy the Python installation from the builder
COPY --from=builder --chown=python:python /python /python

# Copy the application from the builder (already installed deps in .venv)
COPY --from=builder --chown=nonroot:nonroot /app /app

# Place the virtualenv executables at the front of PATH
ENV PATH="/app/.venv/bin:$PATH"

USER nonroot
WORKDIR /app

# Default: run the FastAPI app via fastapi CLI
EXPOSE 8000
CMD ["fastapi", "run", "--host", "0.0.0.0", "main:app"]
