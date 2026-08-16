# syntax=docker/dockerfile:1.10
#
# groundkit service image (SPEC.md §9, Phase 6; ADR-0021).
#
# Build from the REPO ROOT, not from this directory — the build context is the
# project, and `.dockerignore` at the root is what keeps it small:
#
#     docker build -f infra/docker/Dockerfile -t groundkit:local .
#
# Two stages sharing one base tag. That sharing is load-bearing rather than
# tidy: a uv-created virtualenv records the absolute path of the interpreter it
# was built against, so the runtime stage must offer that same interpreter at
# that same path. Bumping PYTHON_VERSION changes both stages together; changing
# only one produces an image whose `grk` cannot start.

ARG PYTHON_VERSION=3.11
# Pinned to the uv release this repo's uv.lock was produced with, so the
# resolver that reads the lock is the resolver that wrote it.
ARG UV_VERSION=0.11.23

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv


# -- builder ---------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim-bookworm AS builder

COPY --from=uv /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    # The base image's interpreter is the one the runtime stage will have.
    # Without this, uv may download its own managed build and the venv would
    # point at a path that does not exist in the final image.
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/groundkit

WORKDIR /src

# Dependency layer first, from the lock and the manifest alone, so editing a
# source file does not invalidate the resolve. README.md and LICENSE are copied
# with them because pyproject.toml declares both (`readme`, `license-files`) and
# hatchling reads them when the project itself is built in the next step.
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project --extra dense --extra otel

# Then the project. `--no-editable` installs a real copy into site-packages, so
# the runtime stage needs no source tree on the path and `/src` is discarded
# with this stage.
#
# `--extra dense` and NOT `--extra rerank`: ADR-0021 decision 4. The compose
# topology includes Ollama, so the vector store has to be present for that
# topology to mean anything; torch is multiple gigabytes for a capability
# `Retriever.search` cannot reach at all (ADR-0012 decision 2).
#
# `--extra otel`: ADR-0022 decision 1. `opentelemetry-api` is already a base
# dependency, so every instrumentation site works with no extra installed at
# all — spans are just non-recording. This image is meant to actually ship
# them, so it carries the SDK and OTLP exporter that make that true; the
# compose stack's collector is what it exports to (ADR-0022 decision 2,
# docker-compose.yml).
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable --extra dense --extra otel


# -- runtime ---------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

# uid/gid 10001 is part of the image's public contract (ADR-0021 decision 3):
# a Kubernetes runAsUser/fsGroup, the ownership Docker copies onto a fresh named
# volume from the image's mountpoint, and a host bind mount all have to name the
# same identity. A uid that moves between rebuilds turns each of those into a
# permission failure on a volume that was writable yesterday.
#
# Not `--system`: that flag's whole job is to allocate from the system uid range,
# and this build allocates explicitly. Passing both makes useradd warn on every
# build that 10001 exceeds SYS_UID_MAX, which reads like a problem and is not.
RUN groupadd --gid 10001 groundkit \
 && useradd --uid 10001 --gid 10001 \
      --home-dir /home/groundkit --create-home --shell /usr/sbin/nologin groundkit

COPY --from=builder --chown=10001:10001 /opt/groundkit /opt/groundkit

# Created here, chowned here: Docker copies a mountpoint's ownership from the
# image onto a *fresh* named volume, which is the only mechanism that gets a
# writable volume for a non-root user without an entrypoint that runs as root
# first. Kubernetes ignores this and uses `fsGroup` instead.
RUN mkdir -p /data/index /data/corpus && chown -R 10001:10001 /data

ENV PATH="/opt/groundkit/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    # The venv was byte-compiled at build time (UV_COMPILE_BYTECODE above), so
    # there is nothing left to write — which is what lets the root filesystem be
    # mounted read-only (ADR-0021 decision 2) without a __pycache__ write
    # failing at import.
    PYTHONDONTWRITEBYTECODE=1 \
    # ADR-0022 decision 4: JSON is opt-in, and a deployment opts in. A local
    # `grk search` in a terminal keeps human-readable output. Read by the
    # formatter that lands with the instrumentation change; harmless before it.
    GROUNDKIT_LOG_FORMAT=json

# A read-only root filesystem needs every writable path named. There are exactly
# two, and both are tmpfs/emptyDir in the deployment manifests. If a third ever
# appears the container fails loudly, which is the intended behaviour.
VOLUME ["/tmp"]

USER 10001:10001
WORKDIR /home/groundkit
EXPOSE 8765

# Uses the venv interpreter rather than curl, which slim does not carry and
# which would be one more package in the attack surface for one HTTP GET.
# `/v1/collections` is a real operation, not a static handler — see
# docs/specs/phase-6-iac-observability.md §4.2 for what it does and does not
# prove.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD ["/opt/groundkit/bin/python", "-c", \
       "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/v1/collections', timeout=4).read()"]

ENTRYPOINT ["grk"]

# `--host 0.0.0.0 --allow-remote-access` is deliberate and is the most
# surprising thing in this file (ADR-0021 decision 1). A process bound to
# 127.0.0.1 inside a container is reachable from nothing — not the host, not a
# sibling container, not a published port — so the bind guard's guarantee has to
# move outward. It is re-established by the publish boundary: compose publishes
# 127.0.0.1:8765:8765, Kubernetes uses a ClusterIP Service, and the Terraform
# module opens no ingress at all (ADR-0020 decision 2).
#
# The consequence, stated where someone about to shorten the command will read
# it: `docker run -p 8765:8765` publishes an UNAUTHENTICATED, content-bearing
# surface — document text and absolute source paths — on every interface of the
# host. So does `--network host`. This image cannot tell those cases apart from
# the safe one; the manifests are what make the guarantee true.
CMD ["serve", \
     "--index-dir", "/data/index", \
     "--base-dir", "/data/corpus", \
     "--host", "0.0.0.0", \
     "--port", "8765", \
     "--allow-remote-access"]
