# Copyright 2024 D-Wave
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

from __future__ import annotations

from typing import NamedTuple, Union

import dash
from dash import MATCH, ctx
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from data import DataSet
from demo_interface import generate_problem_details_table_rows
from src.demo_enums import SolverType
from src.utils import draw_bar_chart, draw_accuracy_bars


@dash.callback(
    Output({"type": "to-collapse-class", "index": MATCH}, "className"),
    inputs=[
        Input({"type": "collapse-trigger", "index": MATCH}, "n_clicks"),
        State({"type": "to-collapse-class", "index": MATCH}, "className"),
    ],
    prevent_initial_call=True,
)
def toggle_left_column(collapse_trigger: int, to_collapse_class: str) -> str:
    """Toggles a 'collapsed' class that hides and shows some aspect of the UI.

    Args:
        collapse_trigger (int): The (total) number of times a collapse button has been clicked.
        to_collapse_class (str): Current class name of the thing to collapse, 'collapsed' if not
            visible, empty string if visible.

    Returns:
        str: The new class name of the thing to collapse.
    """

    classes = to_collapse_class.split(" ") if to_collapse_class else []
    if "collapsed" in classes:
        classes.remove("collapsed")
        return " ".join(classes)
    return to_collapse_class + " collapsed" if to_collapse_class else "collapsed"


@dash.callback(
    Output("features", "max"),
    Output("features", "value"),
    Output("features", "marks"),
    inputs=[
        Input("dataset", "value"),
    ],
)
def create_features_input(data_set: str) -> tuple[int, int, dict]:
    """Updates the max, marks, and default value of the Number of Features slider on load and any
    time the data set is updated.

    Args:
        data_set: The data set selected.

    Returns:
        int: The maximum value for the number of features slider.
        int: The default value for the number of features slider.
        dict: The marks for the slider to display min/max values.
    """
    data = DataSet(data_set)
    max_feat = data.n

    value = round(max_feat / 2)

    marks = {
        1: "1",
        max_feat: str(max_feat),
    }

    return max_feat, value, marks


@dash.callback(
    Output("input-graph", "figure"),
    inputs=[
        Input("input-graph", "hoverData"),
        Input("dataset", "value"),
        Input("input-redund", "value"),
    ],
)
def draw_input_graph(hover_data: dict, data_set: str, show_red: bool) -> go.Figure:
    """Runs on load and any time the data set is updated. Displays the features in the data, with
    a bar showing the relevance of each feature. If show_red is true, then hovering on each
    feature's column shows the correlation between that feature and every other feature.

    Args:
        hover_data: Input information about user mouse location.
        data_set: The data set selected.
        show_red: If we want to see redundancy.

    Returns:
        go.Figure: A Plotly figure object.
    """

    if ctx.triggered_id == "input-graph" and not show_red:
        raise PreventUpdate

    # Load the data set
    data = DataSet(data_set)

    return draw_bar_chart(hover_data, None, data, show_red)


@dash.callback(
    Output("output-graph", "figure"),
    inputs=[
        Input("output-graph", "hoverData"),
        Input("results-redund", "value"),
        Input("selected-features", "data"),
        Input("soln-score", "data"),
        State("dataset", "value"),
    ],
)
def draw_output_graph(
    hover_data: dict, show_red: bool, selected_features: list, soln_score: float, data_set: str
) -> go.Figure:
    """Runs when the optimization step is complete. Displays the same bar graph as on the "Input"
    tab, with selected features solid/heavily outlined and unselected features semi-transparent.

    Args:
        hover_data: Input information about user mouse location.
        show_red: If we want to see redundancy.
        selected_features: The features selected for the model.
        soln_score: The accuracy score for the model using the selected features.
        data_set: The data set selected.

    Returns:
        go.Figure: A Plotly figure object.
    """

    if ctx.triggered_id == "output-graph" and not show_red:
        raise PreventUpdate

    # Load the data set
    data = DataSet(data_set)

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.80, 0.20],
        shared_yaxes=True,
        subplot_titles=("Selected Features", "Accuracy"),
    )

    fig1 = draw_bar_chart(hover_data, selected_features, data, show_red)
    fig2 = draw_accuracy_bars(data, selected_features, soln_score)

    fig.add_trace(fig1["data"][0], row=1, col=1)
    fig.add_trace(fig2["data"][0], row=1, col=2)

    fig.update_xaxes(title_text="Num Features", row=1, col=2)

    # Modify bar chart axis labels:
    fig.update_yaxes(title_text="Feature Relevance to Outcome", row=1, col=1)
    if data.name == "titanic":
        fig.update_xaxes(title_text="Passenger Features", row=1, col=1)
    elif data.name == "scene":
        fig.update_xaxes(title_text="Color and Texture Features in Image", row=1, col=1)

    fig.update_layout(
        showlegend=False,
        yaxis_range=[0, 1.1],
        margin={"t": 30, "l": 0, "b": 0, "r": 0},
    )

    return fig


class RunOptimizationReturn(NamedTuple):
    """Return type for the ``run_optimization`` callback function."""

    score: float = dash.no_update
    features: list = dash.no_update
    problem_details_table: list = dash.no_update
    # Add more return variables here. Return values for callback functions
    # with many variables should be returned as a NamedTuple for clarity.


@dash.callback(
    # The Outputs below must align with `RunOptimizationReturn`.
    Output("soln-score", "data"),
    Output("selected-features", "data"),
    Output("problem-details", "children"),
    background=True,
    inputs=[
        # The first string in the Input/State elements below must match an id in demo_interface.py
        # Remove or alter the following id's to match any changes made to demo_interface.py
        Input("run-button", "n_clicks"),
        State("solver-type-select", "value"),
        State("solver-time-limit", "value"),
        State("features", "value"),
        State("redund", "value"),
        State("dataset", "value"),
    ],
    running=[
        (Output("cancel-button", "className"), "", "display-none"),  # Show/hide cancel button.
        (Output("run-button", "className"), "display-none", ""),  # Hides run button while running.
        (Output("results-tab", "disabled"), True, False),  # Disables results tab while running.
        (Output("results-tab", "label"), "Loading...", "Results"),
        (Output("tabs", "value"), "input-tab", "input-tab"),  # Switch to input tab while running.
        (Output("run-in-progress", "data"), True, False),  # Can block certain callbacks.
    ],
    cancel=[Input("cancel-button", "n_clicks")],
    prevent_initial_call=True,
)
def run_optimization(
    # The parameters below must match the `Input` and `State` variables found
    # in the `inputs` list above.
    run_click: int,
    solver_type: Union[SolverType, int],
    time_limit: float,
    num_features: int,
    redund_penalty: float,
    data_set: str,
) -> RunOptimizationReturn:
    """Runs the optimization and updates UI accordingly.

    This is the main function which is called when the ``Run Optimization`` button is clicked.
    This function takes in all form values and runs the optimization, updates the run/cancel
    buttons, deactivates (and reactivates) the results tab, and draws the output graphs on the
    "Results" tab.

    Args:
        run_click: The (total) number of times the run button has been clicked.
        solver_type: The solver to use for the optimization run defined by SolverType in 
            demo_enums.py.
        time_limit: The solver time limit.
        num_features: The number of features to select.
        redund_penalty: The redundancy penalty selected.
        data_set: The name of the dataset used.

    Returns:
        A NamedTuple (RunOptimizationReturn) containing all outputs to be used when updating the
        HTML template (in ``demo_interface.py``). These are:
            score: The accuracy score for the model built with the selected features.
            features: The features selected.
            problem-details: List of the table rows for the problem details table.
    """

    print("solving...")

    solver_type = SolverType(solver_type)
    solver = "cqm" if solver_type is SolverType.CQM else "nl"

    data = DataSet(data_set)

    solution = data.solve_feature_selection(num_features, 1.0 - redund_penalty, time_limit, solver)

    solution = [int(i) for i in solution]  # Avoid issues with json and int64
    print("solution:", solution)
    score = data.score_indices_cv(solution)

    # Generates a list of table rows for the problem details table.
    problem_details_table = generate_problem_details_table_rows(
        solver=solver_type.label,
        time_limit=time_limit,
    )

    return RunOptimizationReturn(
        score=score,
        features=solution,
        problem_details_table=problem_details_table,
    )
