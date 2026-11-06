"""Events emitted by modules and the bus that delivers them."""

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

ALL_EVENTS = "*"


@dataclass(frozen=True)
class Event:
    """Something a module noticed, e.g. ``motion.start`` or ``gesture.swipe``."""

    type: str
    source: str
    frame_index: int
    timestamp_ms: int
    payload: dict[str, Any] = field(default_factory=dict)


EventHandler = Callable[[Event], None]


class EventBus:
    """Delivers events to handlers subscribed by type, or to ``"*"`` for all events."""

    def __init__(self) -> None:
        self._handlers: defaultdict[str, list[EventHandler]] = defaultdict(list)

    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        self._handlers[event_type].append(handler)

    def publish(self, event: Event) -> None:
        for handler in self._handlers[event.type]:
            handler(event)
        for handler in self._handlers[ALL_EVENTS]:
            handler(event)
