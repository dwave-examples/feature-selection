# Copyright 2025 D-Wave
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pandas as pd
from plotly.colors import sample_colorscale
import plotly.graph_objs as go
from typing import Optional

from data import DataSet
from demo_configs import COLOR_SCALE, GRAPH_FONT_SIZE


def draw_bar_chart(
    hover_data: dict, selected_features: Optional[list], data: DataSet, show_redundancy: bool
) -> go.Figure:
    """Draws the feature relevance bar charts for input/output.

    Args:
        hover_data: Input information about user mouse location.
        selected_features: Solution (if available). If not available then None.
        data: The DataSet object for the given data set.
        show_redundancy: Whether we want to see redundancy.

    Returns:
        go.Figure: A Plotly figure object showing the relevance of each feature.

    """

    df = pd.DataFrame(
        {
            "Feature": data.X.columns,
            "Feature Relevance": data.get_relevance(),
        }
    )

    # Calculate the statistics if showing redundancy
    display_text = []
    hover_cols = {"Feature Relevance": False}

    # When displaying a solution, show selected feature as solid and non-selected
    # features as transparent
    if selected_features:
        opacity = np.repeat(0.3, len(df))
    else:
        opacity = np.repeat(1.0, len(df))
        selected_features = []
    opacity[selected_features] = 1.0

    mlw = np.repeat(1 if data.n < 50 else 0, len(df))
    if data.n < 50:
        mlw[selected_features] = 3

    # Manually calculate the continuous color map to show redundancy.
    # This is required for different opacity levels per bar.
    if hover_data and show_redundancy:
        idx = hover_data["points"][0]["pointIndex"]
        redundancy_data = data.get_redundancy()
        # Protect against case where the last hovered point was from a larger data set.
        if idx < data.n:
            # Note: this modifies global data and so is not compatible with use
            # of the demo in a multi-user environment.  One alternative is to
            # make a copy of the DataFrame prior to modifying the redundancy
            # column.
            df["Redundancy"] = redundancy_data[idx]
            hover_cols["Redundancy"] = False

        color_data = df["Redundancy"].values
        normalized_color_data = (color_data - np.min(color_data)) / (
            np.max(color_data) - np.min(color_data)
        )
        rgba_colors = sample_colorscale(COLOR_SCALE, normalized_color_data, colortype="rgb")

        for i in range(len(rgba_colors)):
            rgba_colors[i] = "rgba" + rgba_colors[i][3:-1] + ", " + str(opacity[i]) + ")"

        if data.n < 30:
            display_text = [round(i.item(), 2) for i in color_data]
    else:
        rgba_colors = [f"rgba(42, 125, 225, {o})" for o in opacity]

    # Plot the bar graph
    fig = go.Figure(
        data=[
            go.Bar(
                x=df["Feature"],
                y=df["Feature Relevance"],
                text=display_text,
                textposition="outside",
            )
        ]
    )

    fig.update_traces(
        marker_color=rgba_colors,
        marker_line_color="black",
        marker_line_width=mlw,
        hoverinfo="none",
        hovertemplate=None,
    )

    fig.update_layout(
        font=dict(size=GRAPH_FONT_SIZE),
        margin={"t": 0, "l": 0, "b": 0, "r": 0},
        yaxis_title="Feature Relevance to Outcome",
        yaxis_range=[0, 1.1],
    )

    # Modify axis labels:
    if data.name == "titanic":
        fig.update_layout(xaxis_title="Passenger Features")
    elif data.name == "scene":
        fig.update_layout(xaxis_title="Color and Texture Features in Image")

    return fig


def draw_accuracy_bars(data: DataSet, selected_features: list, soln_score: float) -> go.Figure:
    """Draws the accuracy bar chart for output.

    Args:
        data: The DataSet object for the given data set.
        selected_features: Solution (if available). If not available then None.
        soln_score: Accuracy score for model using selected_features.

    Returns:
        go.Figure: A Plotly figure object comparing the accuracy of using all features vs only the
        selected features.

    """

    scores = [round(data.baseline_cv_score, 2), round(soln_score, 2)]

    fig = go.Figure(
        data=[
            go.Bar(
                x=[str(data.n), str(len(selected_features))],
                y=scores,
                text=scores,
                textposition="outside",
            )
        ]
    )

    fig.update_traces(
        marker_color=[COLOR_SCALE[1], COLOR_SCALE[-1]],
        marker_line_color="black",
        marker_line_width=1,
        hoverinfo="none",
        hovertemplate=None,
    )

    fig.update_layout(
        font=dict(size=GRAPH_FONT_SIZE),
        margin={"t": 0, "l": 0, "b": 0, "r": 0},
        xaxis_title="Num Features",
        yaxis_range=[0, 1.1],
    )

    return fig
