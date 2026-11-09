"""Write events to a JSONL file: one JSON object per line."""

import json
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Self

from movecore.core.events import Event


class EventLogger:
    """Appends each event as one line, so the log can be followed live with ``tail -f``."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8")

    def write(self, event: Event) -> None:
        record = {
            "logged_at": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "type": event.type,
            "source": event.source,
            "frame_index": event.frame_index,
            "timestamp_ms": event.timestamp_ms,
            "payload": event.payload,
        }
        self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
