from types import SimpleNamespace

from data.dataset_adapter import maybe_apply_keyframe_filter_to_scene_info
from data.manifest_io import save_jsonl


def test_scene_filter(tmp_path):
    keyframe_path = tmp_path / "keyframes.jsonl"
    save_jsonl([
        {"image_name": "0.png"},
        {"image_name": "2.png"},
    ], keyframe_path)

    scene_info = SimpleNamespace(
        train_cameras=[SimpleNamespace(image_name="0.png"), SimpleNamespace(image_name="1.png"), SimpleNamespace(image_name="2.png")],
        test_cameras=[SimpleNamespace(image_name="a.png")],
    )
    args = SimpleNamespace(keyframe_list_path=str(keyframe_path), selection_keep_test=True)
    out = maybe_apply_keyframe_filter_to_scene_info(scene_info, args)
    assert [c.image_name for c in out.train_cameras] == ["0.png", "2.png"]
