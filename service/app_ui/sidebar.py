import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

sidebar = html.Div(
    [   
        dbc.Nav(
            [
                dbc.NavLink("Prediction App", href="/prediction-app", active="exact"),
                dbc.NavLink("Historical Dashboard", href="/dashboard", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)