from __future__ import annotations

import gzip
import os

import pytest

from topodrome.storage.extent_encoding import MAX_EXTENT_BYTES, decode_extent, encode_extent

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("content", [b"", b"small text", b"x" * 8000])
def test_tiny_extents_stay_identity(content):
    assert encode_extent(content) == (content, "identity")


def test_compression_is_deterministic_and_preserves_exact_bytes():
    content = b'{"response":"hello world","index":123}\n' * 10000
    encoded, encoding = encode_extent(content)
    assert encoding == "gzip" and len(encoded) < len(content) / 20
    assert (encoded, encoding) == encode_extent(content)
    assert decode_extent(encoded, encoding, len(content)) == content


def test_incompressible_data_is_not_recompressed():
    content = os.urandom(128 * 1024)
    assert encode_extent(content) == (content, "identity")


def test_a_compressible_header_does_not_force_bad_whole_extent_compression():
    content = b"0" * (32 * 1024) + os.urandom(1024**2)
    assert encode_extent(content) == (content, "identity")


@pytest.mark.parametrize("case", ["truncated", "crc", "trailer", "member", "bomb", "short"])
def test_invalid_gzip_fails_before_publication(case):
    content = b"text" * 10000
    encoded = gzip.compress(content)
    raw_length = len(content)
    if case == "truncated":
        encoded = encoded[:-8]
    elif case == "crc":
        encoded = encoded[:-8] + bytes([encoded[-8] ^ 1]) + encoded[-7:]
    elif case == "trailer":
        encoded += b"trailing bytes"
    elif case == "member":
        encoded += gzip.compress(b"another member")
    elif case == "bomb":
        raw_length = 10
    else:
        raw_length += 1
    with pytest.raises(ValueError):
        decode_extent(encoded, "gzip", raw_length)


@pytest.mark.parametrize("raw_length", [-1, MAX_EXTENT_BYTES + 1])
def test_declared_lengths_are_bounded(raw_length):
    with pytest.raises(ValueError):
        decode_extent(gzip.compress(b"data"), "gzip", raw_length)


def test_identity_lengths_and_encodings_are_explicit():
    assert decode_extent(b"data", "identity", 4) == b"data"
    with pytest.raises(ValueError):
        decode_extent(b"data", "identity", 3)
    with pytest.raises(ValueError):
        decode_extent(b"data", "deflate", 4)
