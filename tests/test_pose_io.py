import json
from pathlib import Path

from data.pose_io import load_pose_records


def test_load_pose_jsonl(tmp_path: Path):
    path = tmp_path / "poses.jsonl"
    row = {"frame_idx": 0, "pose": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    poses = load_pose_records(path)
    assert 0 in poses
    assert poses[0].shape == (4, 4)
