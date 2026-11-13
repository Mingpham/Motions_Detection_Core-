"""What to keep while recording a gesture; the camera loop lives in scripts/record_gestures.py."""

from movecore.hands.dataset import SampleWriter
from movecore.hands.landmarks import Hand

# Keep every 3rd frame (~10 per second): neighbouring frames are near-copies, so spacing
# samples out gives more variety for the same count.
SAMPLE_EVERY = 3


class GestureRecorder:
    """Writes one hand per kept frame while recording is on, until ``target`` samples.

    Frames with no hand or with two hands are skipped: with two, it's unclear which one is
    showing the gesture.
    """

    def __init__(self, writer: SampleWriter, target: int) -> None:
        self.writer = writer
        self.target = target
        self.count = 0
        self.recording = False
        self._frames = 0

    @property
    def done(self) -> bool:
        return self.count >= self.target

    def toggle(self) -> None:
        self.recording = not self.recording and not self.done

    def offer(self, hands: list[Hand]) -> bool:
        """Consider one frame's hands; return True if a sample was written."""
        if not self.recording or len(hands) != 1:
            return False
        self._frames += 1
        if self._frames % SAMPLE_EVERY != 1:
            return False
        self.writer.write(hands[0])
        self.count += 1
        if self.done:
            self.recording = False
        return True

    def status(self, hands: list[Hand]) -> str:
        progress = f"{self.writer.label}: {self.count}/{self.target}"
        if self.done:
            return f"{progress} done - q to quit"
        if len(hands) > 1:
            return f"{progress} - show one hand only"
        if self.recording:
            return f"{progress} recording - SPACE pauses"
        return f"{progress} - SPACE to record, q to quit"
