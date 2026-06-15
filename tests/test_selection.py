from selection.pose_threshold import PoseThresholdSelector
from selection.uniform import UniformSelector


def test_uniform_selector_basic():
    records = [{"frame_idx": i, "image_name": f"{i}.png"} for i in range(10)]
    sel, dbg = UniformSelector(stride=3).select(records)
    assert sel == [0, 3, 6, 9]
    assert len(dbg) == 10


def test_pose_selector_first_frame_kept():
    records = [{"frame_idx": 0, "image_name": "0.png", "pose": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}]
    sel, dbg = PoseThresholdSelector().select(records)
    assert sel == [0]
    assert dbg[0]["reason"] == "first"
