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

"""This file stores the Dash HTML layout for the app."""
from __future__ import annotations
from enum import EnumMeta

from dash import dcc, html
import dash_mantine_components as dmc

from demo_configs import (
    DESCRIPTION,
    MAIN_HEADER,
    SHOW_REDUNDANCY,
    NFEATURES,
    REDUNDANCY,
    SOLVER_TIME,
    THUMBNAIL,
    DATA_SETS,
)
from src.demo_enums import SolverType

THEME_COLOR = "#2d4376"


def slider(label: str, id: str, config: dict) -> html.Div:
    """Slider element for value selection.

    Args:
        label: The title that goes above the slider.
        id: A unique selector for this element.
        config: A dictionary of slider configurations, see dcc.Slider Dash docs.
    """
    return html.Div(
        className="slider-wrapper",
        children=[
            html.Label(label, htmlFor=id),
            dmc.Slider(
                id=id,
                className="slider",
                **config,
                marks=[
                    {"value": config["min"], "label": f'{config["min"]}'},
                    {"value": config["max"], "label": f'{config["max"]}'},
                ],
                labelAlwaysOn=True,
                thumbLabel=f"{label} slider",
                color=THEME_COLOR,
            ),
        ],
    )


def dropdown(label: str, id: str, options: list) -> html.Div:
    """Dropdown element for option selection.

    Args:
        label: The title that goes above the dropdown.
        id: A unique selector for this element.
        options: A list of dictionaries of labels and values.
    """
    return html.Div(
        className="dropdown-wrapper",
        children=[
            html.Label(label, htmlFor=id),
            dmc.Select(
                id=id,
                data=options,
                value=options[0]["value"],
                allowDeselect=False,
            ),
        ],
    )


def checkbox(label: str, id: str, checked: bool) -> html.Div:
    """Checkbox element.

    Args:
        label: The title that goes above the checkbox.
        id: A unique selector for this element.
        checked: Whether the checkbox is checked or not.
    """
    return html.Div(
        className="checklist-wrapper",
        children=[
            dmc.Checkbox(
                id=id,
                label=label,
                checked=checked,
                color=THEME_COLOR,
            )    
        ],
    )


def generate_options(options: list | EnumMeta) -> list[dict]:
    """Generates options for dropdowns, checklists, radios, etc."""
    if isinstance(options, EnumMeta):
        return [
            {"label": option.label, "value": f"{option.value}"} for option in options
        ]

    return [{"label": option, "value": f"{option}"} for option in options]


def generate_settings_form() -> html.Div:
    """This function generates settings for selecting the scenario, model, and solver.

    Returns:
        html.Div: A Div containing the settings for selecting the scenario, model, and solver.
    """

    solver_options = generate_options(SolverType)

    return html.Div(
        className="settings",
        children=[
            dropdown(
                label="Data Set",
                id="dataset",
                options=DATA_SETS,
            ),
            slider(
                label="Number of Features",
                id="num-features",
                config=NFEATURES,
            ),
            slider(
                label="Redundancy Penalty",
                id="redundancy-penalty",
                config=REDUNDANCY,
            ),
            dropdown(
                "Solver",
                "solver-type-select",
                sorted(solver_options, key=lambda op: op["value"]),
            ),
            html.Label("Solver Time Limit (seconds)", htmlFor="solver-time-limit"),
            dmc.NumberInput(
                id="solver-time-limit",
                type="number",
                **SOLVER_TIME,
            ),
        ],
    )


def generate_run_buttons() -> html.Div:
    """Run and cancel buttons to run the optimization."""
    return html.Div(
        id="button-group",
        children=[
            html.Button("Run Optimization", id="run-button", className="button"),
            html.Button(
                "Cancel Optimization",
                id="cancel-button",
                className="button",
                style={"display": "none"},
            ),
        ],
    )


def create_interface():
    """Set the application HTML."""
    return html.Div(
        id="app-container",
        children=[
            html.A(  # Skip link for accessibility
                "Skip to main content",
                href="#main-content",
                id="skip-to-main",
                className="skip-link",
                tabIndex=1,
            ),
            # Below are any temporary storage items, e.g., for sharing data between callbacks.
            dcc.Store(id="run-in-progress", data=False),  # Indicates whether run is in progress
            dcc.Store(id="selected-features", data=[]),
            dcc.Store(id="solution-score", data=0.0),
            # Settings and results columns
            html.Main(
                className="columns-main",
                id="main-content",
                children=[
                    # Left column
                    html.Div(
                        id={"type": "to-collapse-class", "index": 0},
                        className="left-column",
                        children=[
                            html.Div(
                                className="left-column-layer-1",  # Fixed width Div to collapse
                                children=[
                                    html.Div(
                                        className="left-column-layer-2",  # Padding and content wrapper
                                        children=[
                                            html.Div(
                                                [
                                                    html.H1(MAIN_HEADER),
                                                    html.P(DESCRIPTION),
                                                ],
                                                className="title-section",
                                            ),
                                            html.Div(
                                                [
                                                    html.Div(
                                                        html.Div(
                                                            [
                                                                generate_settings_form(),
                                                                generate_run_buttons(),
                                                            ],
                                                            className="settings-and-buttons",
                                                        ),
                                                        className="settings-and-buttons-wrapper",
                                                    ),
                                                    # Left column collapse button
                                                    html.Div(
                                                        html.Button(
                                                            id={
                                                                "type": "collapse-trigger",
                                                                "index": 0,
                                                            },
                                                            className="left-column-collapse",
                                                            title="Collapse sidebar",
                                                            children=[
                                                                html.Div(className="collapse-arrow")
                                                            ],
                                                            **{"aria-expanded": "true"},
                                                        ),
                                                    ),
                                                ],
                                                className="form-section",
                                            ),
                                        ],
                                    )
                                ],
                            ),
                        ],
                    ),
                    # Right column
                    html.Div(
                        className="right-column",
                        children=[
                            dmc.Tabs(
                                id="tabs",
                                value="input-tab",
                                color="white",
                                children=[
                                    html.Header(
                                        className="banner",
                                        children=[
                                            html.Nav(
                                                [
                                                    dmc.TabsList(
                                                        [
                                                            dmc.TabsTab("Input", value="input-tab"),
                                                            dmc.TabsTab(
                                                                "Results",
                                                                value="results-tab",
                                                                id="results-tab",
                                                                disabled=True,
                                                            ),
                                                        ]
                                                    ),
                                                ]
                                            ),
                                            html.Img(src=THUMBNAIL, alt="D-Wave logo"),
                                        ],
                                    ),
                                    dmc.TabsPanel(
                                        value="input-tab",
                                        tabIndex="12",
                                        children=[
                                            html.Div(
                                                className="tab-content-wrapper",
                                                children=[
                                                    html.Div([
                                                        checkbox(
                                                            SHOW_REDUNDANCY,
                                                            "input-redundancy",
                                                            False,
                                                        ),
                                                        dcc.Loading(
                                                            parent_className="input",
                                                            type="circle",
                                                            color=THEME_COLOR,
                                                            delay_show=150,
                                                            children=html.Div(
                                                                [
                                                                    dcc.Graph(
                                                                        id="input-graph",
                                                                        responsive=True,
                                                                        config={"displayModeBar": False},
                                                                    )
                                                                ],
                                                                className="graph",
                                                            ),
                                                        ),
                                                    ], className="tab-content"),
                                                ]
                                            )
                                        ],
                                    ),
                                    dmc.TabsPanel(
                                        value="results-tab",
                                        tabIndex="13",
                                        children=[
                                            html.Div(
                                                className="tab-content-wrapper",
                                                children=[
                                                    html.Div([
                                                        checkbox(
                                                            SHOW_REDUNDANCY,
                                                            "results-redundancy",
                                                            False,
                                                        ),
                                                        dcc.Loading(
                                                            parent_className="results",
                                                            type="circle",
                                                            color=THEME_COLOR,
                                                            delay_show=150,
                                                            children=html.Div(
                                                                [
                                                                    dcc.Graph(
                                                                        id="output-graph",
                                                                        responsive=True,
                                                                        config={
                                                                            "displayModeBar": False
                                                                        },
                                                                    ),
                                                                ],
                                                                className="graph",
                                                            ),
                                                        ),
                                                    ], className="tab-content"),
                                                ],
                                            )
                                        ],
                                    ),
                                ],
                            )
                        ],
                    ),
                ],
            ),
        ],
    )
