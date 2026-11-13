import numpy as np

from movecore.hands.landmarks import Hand
from movecore.hands.recorder import SAMPLE_EVERY, GestureRecorder


class FakeWriter:
    label = "rock"

    def __init__(self):
        self.rows = []

    def write(self, hand):
        self.rows.append(hand)


HAND = Hand("Right", 0.9, np.zeros((21, 3)), np.zeros((21, 3)))


def test_nothing_is_written_until_recording_starts():
    recorder = GestureRecorder(FakeWriter(), target=5)

    assert not recorder.offer([HAND])
    assert "SPACE to record" in recorder.status([HAND])


def test_every_few_frames_are_kept_until_the_target():
    writer = FakeWriter()
    recorder = GestureRecorder(writer, target=3)
    recorder.toggle()

    kept = [recorder.offer([HAND]) for _ in range(20)]

    assert len(writer.rows) == 3
    assert kept[0] and kept[SAMPLE_EVERY] and not kept[1]
    assert recorder.done and not recorder.recording
    assert "3/3 done" in recorder.status([HAND])


def test_frames_with_no_hand_or_two_hands_are_skipped():
    writer = FakeWriter()
    recorder = GestureRecorder(writer, target=5)
    recorder.toggle()

    recorder.offer([])
    recorder.offer([HAND, HAND])

    assert writer.rows == []
    assert "one hand only" in recorder.status([HAND, HAND])


def test_space_pauses_and_cannot_restart_when_done():
    recorder = GestureRecorder(FakeWriter(), target=1)
    recorder.toggle()
    recorder.toggle()
    assert not recorder.recording

    recorder.toggle()
    recorder.offer([HAND])
    recorder.toggle()
    assert recorder.done and not recorder.recording
