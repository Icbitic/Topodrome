"""Optional HTTP-body compression; offsets and hashes always describe raw bytes.

Only upload extent bodies are encoded. Credentials and HTTP headers never enter
the compression context, and the authority verifies the decoded extent before
writing any bytes. Every gzip body is one complete, independently bounded member.
"""

from __future__ import annotations

import gzip
import zlib

MIN_COMPRESS_BYTES = 16 * 1024
SAMPLE_BYTES = 32 * 1024
MIN_SAVING_BYTES = 4 * 1024
MAX_EXTENT_BYTES = 8 * 1024**2


def encode_extent(content: bytes) -> tuple[bytes, str]:
    """Skip tiny/already-compressed data; use cheap deterministic gzip otherwise."""
    if len(content) < MIN_COMPRESS_BYTES:
        return content, "identity"
    sample = content[:SAMPLE_BYTES]
    trial = gzip.compress(sample, compresslevel=1, mtime=0)
    if len(trial) * 10 > len(sample) * 9:
        return content, "identity"
    encoded = (
        trial if len(sample) == len(content) else gzip.compress(content, compresslevel=1, mtime=0)
    )
    if len(encoded) * 10 > len(content) * 9 or len(content) - len(encoded) < MIN_SAVING_BYTES:
        return content, "identity"
    return encoded, "gzip"


def decode_extent(content: bytes, encoding: str, raw_length: int) -> bytes:
    """Reject truncation, extra members/trailers and expansion past the extent."""
    if not 0 <= raw_length <= MAX_EXTENT_BYTES or len(content) > MAX_EXTENT_BYTES:
        raise ValueError("upload extent exceeds its memory bound")
    if encoding == "identity":
        if len(content) != raw_length:
            raise ValueError("upload decoded length mismatch")
        return content
    if encoding != "gzip":
        raise ValueError("unsupported upload encoding")
    try:
        decoder = zlib.decompressobj(zlib.MAX_WBITS | 16)
        result = decoder.decompress(content, raw_length + 1)
        if (
            len(result) != raw_length
            or not decoder.eof
            or decoder.unused_data
            or decoder.unconsumed_tail
        ):
            raise ValueError("invalid bounded gzip extent")
        return result
    except zlib.error:
        raise ValueError("invalid gzip extent") from None
