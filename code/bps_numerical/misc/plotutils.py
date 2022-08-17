from typing import List, Tuple

import pandas as pd
from loguru import logger

MATPLOTLIB = True
PLOTLY = True
try:
    import plotly.express as px
except ModuleNotFoundError:
    PLOTLY = False

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    MATPLOTLIB = False


def plot_feature_scores(feature_scores: List[Tuple[str, int]], title=None, **kwargs):
    if not MATPLOTLIB or not PLOTLY:
        logger.warning("MATPLOTLIB/PLOTLY not found!")
        return False

    data = sorted(feature_scores, key=lambda f: f[1])
    data = pd.DataFrame(data, columns=["gene", "score"])
    fig = px.bar(
        data,
        x="score",
        y="gene",
        title=title or "Gene Feature Importances",
        text_auto=True,
        orientation="h",
        height=kwargs.get("height"),
        width=kwargs.get("width"),
    )
    fig.update_layout(
        yaxis=dict(tickfont=dict(size=kwargs.get("font_size", 7))),
        title_font_size=kwargs.get("title_font_size", 10),
    )
    fig.show()
    return fig
