"""User-selected Split direction (M2D candidate lines) is honoured per cut."""
import inspect

from onestring_physics import paper_split_reflatten as R
from onestring_physics.onestring_pipeline import PipelineParameters


def test_direction_choices_parameter_and_planner_hook():
    assert PipelineParameters().paper_split_direction_choices == ()
    sig = inspect.signature(R.plan_reflatten_cuts)
    assert "direction_choices" in sig.parameters
    src = inspect.getsource(R.plan_reflatten_cuts)
    assert 'chosen_by="user"' in src and "options[choice]" in src
