from itertools import islice, pairwise

import pytest

from movecore.capture import SourceError, VideoFileSource


def test_reads_every_frame_with_video_timestamps(make_video):
    with VideoFileSource(make_video(frames=5, fps=10.0)) as source:
        frames = list(source.frames())

    assert [timestamp for _, timestamp in frames] == [0, 100, 200, 300, 400]
    assert frames[0][0].shape == (120, 160, 3)


def test_loop_keeps_timestamps_increasing(make_video):
    with VideoFileSource(make_video(frames=4), loop=True) as source:
        timestamps = [timestamp for _, timestamp in islice(source.frames(), 10)]

    assert len(timestamps) == 10
    assert all(later > earlier for earlier, later in pairwise(timestamps))


def test_missing_file_raises(tmp_path):
    with pytest.raises(SourceError, match="not found"):
        VideoFileSource(tmp_path / "missing.avi")


def test_file_that_is_not_a_video_raises(tmp_path):
    path = tmp_path / "notes.avi"
    path.write_text("not a video")

    with pytest.raises(SourceError, match="Could not open"):
        VideoFileSource(path)


def test_closed_source_yields_nothing(make_video):
    source = VideoFileSource(make_video())
    source.close()

    assert list(source.frames()) == []
