FROM python:3.12-slim-bookworm@sha256:1aaa65a85fda306ffb8b910824d4e93bdce61e212c7e87168123ea3073b41a1a
COPY wheels /tmp/wheels
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-index --find-links=/tmp/wheels -r /tmp/requirements.txt && rm -rf /tmp/wheels /tmp/requirements.txt
RUN useradd -m agent && mkdir -p /app/workspace /logs/artifacts && chown -R agent:agent /app /logs/artifacts
WORKDIR /app/workspace
