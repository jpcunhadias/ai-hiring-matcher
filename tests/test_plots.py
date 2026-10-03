import pandas as pd

from src.plots import render_all


def _sample_df() -> pd.DataFrame:
    rows = []
    for role, (female, male) in {"Nurse": (9, 1), "Pilot": (1, 9), "Chef": (1, 8)}.items():
        # `female`/`male` of every 10 rows per gender get Best Match = 1
        for gender, positives in (("Female", female), ("Male", male)):
            rows += [
                {"Job Roles": role, "Gender": gender, "Best Match": int(i < positives)}
                for i in range(10)
            ]
    return pd.DataFrame(rows)


def test_render_all_writes_a_light_and_dark_png_per_figure(tmp_path):
    written = render_all(_sample_df(), tmp_path)

    assert sorted(p.name for p in written) == [
        "fairness-bimodality-dark.png",
        "fairness-bimodality-light.png",
        "fairness-gap-by-role-dark.png",
        "fairness-gap-by-role-light.png",
    ]
    for path in written:
        assert path.read_bytes().startswith(b"\x89PNG")
