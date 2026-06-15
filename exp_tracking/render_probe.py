"""Future render probe utilities.

Target use-case:
- perturb a selected keyframe pose by small delta SE(3)
- render nearby views
- log image strips and sensitivity metrics
"""


def run_render_probe(*args, **kwargs):
    raise NotImplementedError("Render probe will be implemented after base train/eval is stable.")
