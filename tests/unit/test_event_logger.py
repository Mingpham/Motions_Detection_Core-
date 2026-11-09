import json

from movecore.core.events import Event
from movecore.output import EventLogger

EVENT = Event("zone.enter", "motion", 12, 400, {"zone": "door"})


def read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_each_event_is_one_json_line(tmp_path):
    path = tmp_path / "logs" / "events.jsonl"

    with EventLogger(path) as logger:
        logger.write(EVENT)
        logger.write(EVENT)

    records = read_lines(path)
    assert len(records) == 2
    assert records[0]["type"] == "zone.enter"
    assert records[0]["payload"] == {"zone": "door"}
    assert records[0]["frame_index"] == 12
    assert records[0]["logged_at"].endswith("+00:00")


def test_appends_across_runs(tmp_path):
    path = tmp_path / "events.jsonl"
    for _ in range(2):
        with EventLogger(path) as logger:
            logger.write(EVENT)

    assert len(read_lines(path)) == 2


def test_each_line_is_written_immediately(tmp_path):
    path = tmp_path / "events.jsonl"
    with EventLogger(path) as logger:
        logger.write(EVENT)
        assert len(read_lines(path)) == 1
