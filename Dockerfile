# syntax=docker/dockerfile:1

FROM tailscale/tailscale:stable AS tailscale
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY --from=tailscale /usr/local/bin/tailscale /usr/local/bin/tailscale
COPY --from=tailscale /usr/local/bin/tailscaled /usr/local/bin/tailscaled

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

COPY docker/start-with-tailscale.sh /usr/local/bin/start-with-tailscale

RUN chmod 0755 /usr/local/bin/start-with-tailscale \
    && useradd --create-home --uid 1000 trader \
    && mkdir -p /tmp/tailscale \
    && chown -R trader:trader /tmp/tailscale /app

USER trader

ENTRYPOINT ["/usr/local/bin/start-with-tailscale"]
CMD ["kalshi-bot", "--help"]
