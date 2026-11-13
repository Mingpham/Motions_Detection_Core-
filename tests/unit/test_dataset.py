import numpy as np
import pytest

from movecore.hands.dataset import COLUMNS, DatasetError, SampleWriter, check_label, load_samples
from movecore.hands.landmarks import Hand


def hand(points, side="Right"):
    return Hand(side, 0.9, np.zeros((21, 3)), points)


@pytest.mark.parametrize("label", ["rock", "call_me", "g2"])
def test_plain_labels_are_accepted(label):
    assert check_label(label) == label


@pytest.mark.parametrize("label", ["", "Rock", "../escape", "a/b", "rock.csv", "x" * 33])
def test_labels_that_could_break_file_names_are_rejected(label):
    with pytest.raises(DatasetError, match="Gesture label"):
        check_label(label)


def test_writer_adds_a_header_once_and_appends(tmp_path, hand_points):
    for session in ("s1", "s2"):
        with SampleWriter(tmp_path, "rock", session) as writer:
            writer.write(hand(hand_points()))

    lines = (tmp_path / "rock.csv").read_text().splitlines()
    assert lines[0] == ",".join(COLUMNS)
    assert len(lines) == 3
    assert lines[2].startswith("rock,s2,Right,0,")


def test_a_hand_gives_the_same_features_mirrored_or_not(tmp_path, hand_points):
    # Your right hand: seen directly it has a right-hand shape; in a mirrored image the
    # landmarks are mirrored too, and it has a left-hand shape. Both must give one answer.
    right = hand_points("1100")
    with SampleWriter(tmp_path, "rock", "plain") as writer:
        writer.write(Hand("Right", 0.9, np.zeros((21, 3)), right))
    with SampleWriter(tmp_path, "rock", "mirror") as writer:
        writer.write(Hand("Right", 0.9, np.zeros((21, 3)), right * (-1, 1, 1), mirrored=True))

    plain, mirrored = load_samples(tmp_path).features
    assert mirrored == pytest.approx(plain)


def test_writer_rejects_a_bad_label(tmp_path):
    with pytest.raises(DatasetError):
        SampleWriter(tmp_path, "../rock", "s1")
    assert list(tmp_path.iterdir()) == []


def test_load_turns_rows_into_features(write_gestures):
    dataset = load_samples(write_gestures(samples=10))

    assert len(dataset) == 60  # 3 gestures × 2 sessions × 10
    assert dataset.features.shape == (60, 63)
    assert dataset.counts() == {"fist": 20, "open": 20, "peace": 20}
    assert dataset.session_counts() == {"fist": 2, "open": 2, "peace": 2}
    assert "open/s1" in dataset.sessions


def test_empty_folder_says_how_to_record(tmp_path):
    with pytest.raises(DatasetError, match="record_gestures.py"):
        load_samples(tmp_path)


def test_foreign_csv_is_rejected(tmp_path):
    (tmp_path / "other.csv").write_text("name,value\na,1\n")
    with pytest.raises(DatasetError, match="unexpected columns"):
        load_samples(tmp_path)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"handedness": "Both"}, "handedness"),
        ({"mirrored": "yes"}, "mirrored must be 0 or 1"),
        ({"label": "../x"}, "Gesture label"),
        ({"x5": "nan"}, "x5"),
        ({"y0": "abc"}, "could not convert"),
    ],
)
def test_bad_rows_name_the_file_and_line(tmp_path, hand_points, change, message):
    with SampleWriter(tmp_path, "rock", "s1") as writer:
        writer.write(hand(hand_points()))
    header, row = (tmp_path / "rock.csv").read_text().splitlines()
    values = dict(zip(COLUMNS, row.split(","), strict=True)) | change
    (tmp_path / "rock.csv").write_text(f"{header}\n{','.join(values.values())}\n")

    with pytest.raises(DatasetError, match=f"rock.csv, line 2: .*{message}"):
        load_samples(tmp_path)
