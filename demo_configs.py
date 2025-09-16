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

"""This file stores input parameters for the app."""

# THEME_COLOR is used for the button, text, and banner and should be dark
# and pass accessibility checks with white: https://webaim.org/resources/contrastchecker/
# THEME_COLOR_SECONDARY can be light or dark and is used for sliders, loading icon, and tabs
THEME_COLOR = "#074C91"  # D-Wave dark blue default #074C91
THEME_COLOR_SECONDARY = "#2A7DE1"  # D-Wave blue default #2A7DE1
COLOR_SCALE = ["#074C91", "#2A7DE1", "#17BEBB", "#FFA143", "#F37820"]

GRAPH_FONT_SIZE = 14

THUMBNAIL = "static/dwave_logo.svg"

APP_TITLE = "Feature Selection"
MAIN_HEADER = "Feature Selection"
DESCRIPTION = """\
The goal for this application is to choose features that will help the machine learning model learn
by promoting diversity between features and strong relationships to the target variable.
"""

#######################################
# Sliders, buttons and option entries #
#######################################

# Input sliders
NFEATURES = {
    "min": 1,
    "max": 10,
    "step": 1,
    "value": 5,
}

REDUNDANCY = {
    "min": 0,
    "max": 1,
    "step": 0.1,
    "value": 0.5,
}

# List of datasets for drop down
DATA_SETS = [
    {"label": "Titanic Survival", "value": "titanic"},
    {"label": "Scene", "value": "scene"},
]


SHOW_REDUNDANCY = ["Show Redundancy"]


# solver time limits in seconds (value means default)
SOLVER_TIME = {
    "min": 10,
    "max": 300,
    "step": 5,
    "value": 10,
}
