"""Private, fsynced compute-completion receipts. Never contain fencing tokens."""

import json
import os
from pathlib import Path
from uuid import uuid4

from topodrome.storage.paths import validate_component


class FinalizationJournal:
    def __init__(self, root: Path):
        self.root = root
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        root.chmod(0o700)

    def _path(self, attempt_id):
        return self.root / (validate_component(attempt_id, label="attempt_id") + ".json")

    def save(self, receipt):
        path = self._path(receipt["attempt_id"])
        temporary = path.with_suffix("." + uuid4().hex + ".tmp")
        try:
            with temporary.open("x", encoding="utf-8") as stream:
                temporary.chmod(0o600)
                json.dump(receipt, stream, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            self._sync()
        finally:
            temporary.unlink(missing_ok=True)

    def pending(self):
        for path in sorted(self.root.glob("*.json")):
            if path.is_symlink() or path.stat().st_size > 16 * 1024 * 1024:
                raise RuntimeError("invalid private finalization receipt")
            receipt = json.loads(path.read_text())
            if self._path(receipt["attempt_id"]) != path:
                raise RuntimeError("finalization receipt identity mismatch")
            yield receipt

    def contains(self, attempt_id):
        return self._path(attempt_id).is_file()

    def complete(self, attempt_id):
        self._path(attempt_id).unlink(missing_ok=True)
        self._sync()

    def _sync(self):
        fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
