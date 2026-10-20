"""Minimal stdlib client for RAGMAP's ``ragmap-encoder-service`` (``ragmap.encoders``).

The container cannot import ragmap, so this mirrors ``ragmap.encoders.wire`` and
``ragmap.encoders.client``: images go out as base64 PNG (RGB uint8), embeddings come back
as base64 little-endian float32 ``(n, D)``, already L2-normalised by the service. Every
answer is checked against the service's ``/healthz`` identity (profile, revision, dim) and
every error status raises, so a build can never receive zeros or mix two encoders.
"""
from __future__ import annotations

import base64
import io
import json
import urllib.error
import urllib.request

import numpy as np


class EncoderServiceError(RuntimeError):
    pass


def encode_png(image) -> str:
    """Base64 PNG of a PIL image, converted to RGB (``wire.encode_png``)."""
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG", compress_level=1)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def decode_embeddings(data: str, count: int, dim: int) -> np.ndarray:
    array = np.frombuffer(base64.b64decode(data), dtype="<f4")
    if array.size != count * dim:
        raise EncoderServiceError(f"embedding payload has {array.size} floats, expected {count} x {dim}")
    return array.reshape(count, dim).astype(np.float32)


class EncoderClient:
    def __init__(self, base_url: str, profile: str = "", timeout: float = 600.0):
        if not base_url:
            raise EncoderServiceError("ragmap.encoder=http needs ragmap.encoder_service_url")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.health = self._call("/healthz")
        if self.health.get("status") != "ok":
            raise EncoderServiceError(f"{self.base_url} is not ready: {self.health}")
        if profile and self.health.get("profile") != profile:
            raise EncoderServiceError(
                f"{self.base_url} serves profile {self.health.get('profile')!r}, not ragmap.encoder_profile={profile!r}")
        if not self.health.get("normalized", False):
            raise EncoderServiceError(f"{self.base_url} does not return L2-normalised embeddings: {self.health}")
        self.dim = int(self.health["dim"])

    def _call(self, path: str, payload=None) -> dict:
        data = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(self.base_url + path, data=data,
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            raise EncoderServiceError(f"{self.base_url}{path}: HTTP {exc.code}: {detail}") from None
        except urllib.error.URLError as exc:
            raise EncoderServiceError(f"{self.base_url}{path}: {exc.reason}") from None

    def embed_images(self, images) -> np.ndarray:
        """PIL images -> ``(n, D)`` float32, L2-normalised by the service."""
        answer = self._call("/v1/embed/images", {"images": [encode_png(image) for image in images]})
        for field in ("profile", "revision", "dim"):
            if answer.get(field) != self.health[field]:
                raise EncoderServiceError(f"{field} changed under us: {answer.get(field)} != {self.health[field]}")
        if answer.get("count") != len(images) or answer.get("dtype") != "float32":
            raise EncoderServiceError(f"bad answer for {len(images)} images: count={answer.get('count')}, "
                                      f"dtype={answer.get('dtype')}")
        return decode_embeddings(answer["embeddings"], len(images), self.dim)

    def identity(self) -> dict:
        return {key: self.health.get(key) for key in ("profile", "model", "revision", "dim", "normalized")}
