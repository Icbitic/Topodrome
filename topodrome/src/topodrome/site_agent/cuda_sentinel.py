"""Minimal CUDA-context process used for cooperative scavenger handoff.

This module is launched as a subprocess with one physical GPU exposed through
``CUDA_VISIBLE_DEVICES``. It allocates no tensor memory and waits for its parent
to close stdin.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import sys


def _initialize_context() -> None:
    names = tuple(
        name
        for name in (
            ctypes.util.find_library("cudart"),
            "libcudart.so",
            "libcudart.so.12",
        )
        if name
    )
    last_error: BaseException | None = None
    for name in names:
        try:
            library = ctypes.CDLL(name)
            library.cudaSetDevice.argtypes = [ctypes.c_int]
            library.cudaSetDevice.restype = ctypes.c_int
            library.cudaFree.argtypes = [ctypes.c_void_p]
            library.cudaFree.restype = ctypes.c_int
            set_device = int(library.cudaSetDevice(0))
            initialize = int(library.cudaFree(None))
            if set_device != 0 or initialize != 0:
                raise RuntimeError(
                    f"CUDA context initialization failed ({set_device}, {initialize})"
                )
            return
        except (AttributeError, OSError, RuntimeError) as exc:
            last_error = exc
    raise RuntimeError("CUDA runtime library is unavailable") from last_error


def main() -> int:
    try:
        _initialize_context()
    except Exception as exc:
        print(f"ERROR {type(exc).__name__}: {exc}", flush=True)
        return 1
    print("READY", flush=True)
    sys.stdin.buffer.read(1)
    return 0


if __name__ == "__main__":  # pragma: no cover - subprocess entrypoint
    raise SystemExit(main())
