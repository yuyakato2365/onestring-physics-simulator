"""UI defaults/labels for the 2026-09-13 surface-constrained K3D experiment.

This patch deliberately changes display/defaults only.  Internally the mature
``optcuts_test2`` route remains the numerical carrier so existing wrappers can
continue keying off ONESTRING_OPTCUTS_TEST_VARIANT=2.
"""
from __future__ import annotations

from typing import Any


VERSION_ID = "2026-09-13-surface-constrained-k3d"
VERSION_LABEL = "2026-09-13 K3D Surface-Constrained Planarization"
VERSION_DESCRIPTION = (
    "OptCuts-test2系を土台に、K3Dの各頂点を元の入力メッシュ表面へ最近傍投影しながら、"
    "quad平面化との交互射影を行う実験版。K2Dはglobal all-hinge Phase 1を使用します。"
)
OMEGA_TEST2_LABEL = "2026-09-13 Surface-Constrained K3D + Global All-Hinge K2D"


def install_20260913_ui_patch(st: Any) -> None:
    if getattr(st, "_onestring_20260913_ui_patch_installed", False):
        return

    original_selectbox = st.selectbox
    original_number_input = st.number_input

    def patched_selectbox(label: str, options: Any, *args: Any, **kwargs: Any):
        option_list = list(options)

        # The top-most legacy version selector receives a new preserved version.
        if label == "version" and option_list and isinstance(option_list[0], dict):
            if not any(str(item.get("id", "")) == VERSION_ID for item in option_list if isinstance(item, dict)):
                base = dict(option_list[0])
                base.update(
                    {
                        "id": VERSION_ID,
                        "label": VERSION_LABEL,
                        "description": VERSION_DESCRIPTION,
                    }
                )
                option_list.insert(0, base)
            kwargs["index"] = 0
            return original_selectbox(label, option_list, *args, **kwargs)

        # app_optcuts.py still routes the internal string optcuts_test2.  We keep
        # that stable identifier but show the dated implementation name instead.
        if label == "Omega parameterization mode" and "optcuts_test2" in option_list:
            prior_format = kwargs.get("format_func")

            def format_mode(value: Any) -> str:
                if value == "optcuts_test2":
                    return OMEGA_TEST2_LABEL
                if prior_format is not None:
                    return str(prior_format(value))
                return str(value)

            kwargs["format_func"] = format_mode
            kwargs["index"] = option_list.index("optcuts_test2")
            return original_selectbox(label, option_list, *args, **kwargs)

        return original_selectbox(label, options, *args, **kwargs)

    def patched_number_input(label: str, *args: Any, **kwargs: Any):
        if label == "tile size":
            kwargs["value"] = 0.1
        return original_number_input(label, *args, **kwargs)

    st.selectbox = patched_selectbox
    st.number_input = patched_number_input
    st._onestring_20260913_ui_patch_installed = True


__all__ = [
    "install_20260913_ui_patch",
    "VERSION_ID",
    "VERSION_LABEL",
    "OMEGA_TEST2_LABEL",
]
