# syntax=docker/dockerfile:1.7
# Simulated backend only: no torch, no model weights.

FROM python:3.13-slim AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /src
COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m venv /opt/venv && /opt/venv/bin/pip install '.[speed]'

FROM python:3.13-slim
COPY --from=build /opt/venv /opt/venv
RUN useradd --system --uid 10001 --no-create-home tokenserve
USER tokenserve
# Loopback inside a container is unreachable from the host, so the image
# listens on all interfaces. Publish the admin port only where it is needed,
# e.g. `docker run -p 8000:8000 -p 127.0.0.1:9090:9090 tokenserve:dev`.
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    TOKENSERVE_HOST=0.0.0.0 \
    TOKENSERVE_ADMIN_HOST=0.0.0.0
EXPOSE 8000 9090
HEALTHCHECK --interval=10s --timeout=3s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:9090/readyz', timeout=2)"]
ENTRYPOINT ["tokenserve"]
