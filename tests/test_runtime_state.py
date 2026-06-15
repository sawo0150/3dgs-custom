from core.state import SessionState


def test_session_state_defaults():
    state = SessionState(session_id="abc")
    assert state.current_map_version == 0
    assert state.keyframe_ids == []
