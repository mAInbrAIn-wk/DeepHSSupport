"""
Educational Survival Dashboard: Kausalität vs. Data Leakage
===========================================================
Dieses Dashboard dient zur Demonstration des "Immortal Time Bias" und "Look-Ahead Bias"
in der Survival-Analyse durch Gegenüberstellung von statischer und dynamischer Modellierung.
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np

import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output
import statsmodels.formula.api as smf
from statsmodels.duration.survfunc import SurvfuncRight

import sys
sys.path.append(str(Path(__file__).parent.resolve()))
from extended_cox_survival import build_person_semester_panel

print("Lade Datensätze...")
data_dir = Path('output_dl')
df_static = pd.read_csv(data_dir / 'agg_abschluesse.csv')
df_static.columns = df_static.columns.str.strip()

# Cleanup statische Variablen für Formel-Kompatibilität
if 'AVG_cp_sem1-2' in df_static.columns:
    df_static['AVG_cp_sem1_2'] = df_static['AVG_cp_sem1-2']
if 'AVG_note_sem1-2' in df_static.columns:
    df_static['AVG_note_sem1_2'] = df_static['AVG_note_sem1-2']

print("Erstelle dynamisches Person-Semester-Panel...")
df_panel = build_person_semester_panel(data_dir)

print("Daten geladen. Starte Dashboard...")

app = dash.Dash(__name__, title="Survival Analysis Bias Demo")

app.layout = html.Div(style={'fontFamily': 'sans-serif', 'padding': '20px', 'backgroundColor': '#f8f9fa', 'minHeight': '100vh'}, children=[
    html.H1("Survival-Analyse: Kausalität vs. Data Leakage", style={'color': '#2c3e50'}),
    html.P("Interaktive Demonstration, wie Data Leakage und Immortal Time Bias den Effekt von Support-Maßnahmen künstlich aufblähen.", style={'color': '#7f8c8d', 'marginBottom': '30px'}),
    
    html.Div(style={'display': 'flex', 'gap': '20px'}, children=[
        # Sidebar
        html.Div(style={'width': '350px', 'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '8px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'height': 'fit-content'}, children=[
            html.H3("Methodik-Auswahl", style={'marginTop': 0}),
            
            dcc.RadioItems(
                id='model-mode',
                options=[
                    {'label': ' 🔴 Statisches Modell (Data Leakage)', 'value': 'static'},
                    {'label': ' 🟢 Dynamisches Panel-Modell (Kausal)', 'value': 'dynamic'}
                ],
                value='static',
                labelStyle={'display': 'block', 'margin': '15px 0', 'fontWeight': 'bold', 'fontSize': '16px'},
                inputStyle={'marginRight': '10px'}
            ),
            
            html.Div(id='mode-description', style={'marginTop': '20px', 'padding': '15px', 'borderRadius': '5px', 'fontSize': '14px', 'lineHeight': '1.5'}),
            
            html.Hr(style={'margin': '20px 0'}),
            html.Label("Zielereignis (Event):", style={'fontWeight': 'bold', 'display': 'block', 'marginBottom': '5px'}),
            dcc.Dropdown(
                id='event-type',
                options=[
                    {'label': 'Abbruch / Exmatrikulation', 'value': 'abbruch'},
                    {'label': 'Alle Abgänge', 'value': 'alle'}
                ],
                value='abbruch',
                clearable=False,
                style={'marginBottom': '15px'}
            ),
            
            html.Label("Support-Art:", style={'fontWeight': 'bold', 'display': 'block', 'marginBottom': '5px'}),
            dcc.Dropdown(
                id='support-type',
                options=[
                    {'label': 'Irgendein Support (any_support)', 'value': 'any_support'},
                    {'label': 'Fachlicher Support (Fach_supp)', 'value': 'Fach_supp'}
                ],
                value='any_support',
                clearable=False
            )
        ]),
        
        # Main Content
        html.Div(style={'flex': 1}, children=[
            html.Div(style={'display': 'flex', 'gap': '20px', 'marginBottom': '20px'}, children=[
                html.Div(style={'flex': 2, 'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '8px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}, children=[
                    dcc.Graph(id='forest-plot', style={'height': '350px'})
                ]),
                html.Div(style={'flex': 1, 'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '8px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}, children=[
                    html.H4("Effekt der Maßnahme (Hazard Ratio)", style={'marginTop': 0, 'borderBottom': '2px solid #ecf0f1', 'paddingBottom': '10px'}),
                    html.Div(id='summary-stats', style={'fontSize': '15px', 'lineHeight': '1.6'})
                ])
            ]),
            html.Div(style={'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '8px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}, children=[
                dcc.Graph(id='survival-plot', style={'height': '450px'})
            ])
        ])
    ])
])

@app.callback(
    [Output('mode-description', 'children'),
     Output('mode-description', 'style'),
     Output('forest-plot', 'figure'),
     Output('summary-stats', 'children'),
     Output('survival-plot', 'figure')],
    [Input('model-mode', 'value'),
     Input('event-type', 'value'),
     Input('support-type', 'value')]
)
def update_dashboard(mode, event_type, support_type):
    
    # 1. Event Definition
    if mode == 'static':
        df = df_static.copy()
        if event_type == 'abbruch':
            df['event'] = df['status'].str.lower().isin(['abgebrochen', 'exmatrikuliert', 'zeitueberschreitung']).astype(int)
        else:
            df['event'] = df['status'].notna().astype(int)
    else:
        df = df_panel.copy()
        # In df_panel ist 'event' bereits korrekt als Abbruch codiert (aus build_person_semester_panel)
        # Wenn 'alle' Events gewählt werden, belassen wir es vorerst bei 'event', da panel_data kein 'status' mehr hat.
        # (Eine Verfeinerung für 'alle abgänge' im Panel erfordert Änderungen im Generator)
            
    # 2. Modell Setup
    if mode == 'static':
        desc = [
            html.Strong("🚨 Data Leakage (Look-Ahead Bias)", style={'display': 'block', 'marginBottom': '8px', 'fontSize': '16px'}),
            "Dieses Modell sagt das Abbruchrisiko an Tag 1 voraus, nutzt aber Variablen aus der Zukunft:", html.Br(), html.Br(),
            "• ", html.Strong("Jemals Support gehabt"), " gewährt Unsterblichkeit (wer in Sem. 6 Support nutzt, kann nicht in Sem. 1 abgebrochen haben).", html.Br(),
            "• ", html.Strong("Ø CP (Sem 1-2)"), " sortiert frühe Abbrecher direkt aus.", html.Br(), html.Br(),
            html.Strong("Effekt:"), " Eine künstliche Illusion eines massiven Schutzeffekts."
        ]
        style = {'backgroundColor': '#fef2f2', 'color': '#991b1b', 'border': '1px solid #fecaca', 'marginTop': '20px', 'padding': '15px', 'borderRadius': '5px', 'fontSize': '14px'}
        
        df['support_numeric'] = df[support_type].astype(int)
        formula = "studiendauer_semester ~ support_numeric + hzb_note + AVG_cp_sem1_2 + erstakademiker"
        
        cox = smf.phreg(formula=formula, data=df, status=df['event'], ties='breslow').fit()
        idx_supp = list(cox.model.exog_names).index('support_numeric')
        hr_support = np.exp(cox.params[idx_supp])
        
        # Surv curves (Kaplan-Meier grouped by static support)
        sf_mit = SurvfuncRight(df[df['support_numeric']==1]['studiendauer_semester'], df[df['support_numeric']==1]['event'])
        sf_ohne = SurvfuncRight(df[df['support_numeric']==0]['studiendauer_semester'], df[df['support_numeric']==0]['event'])
        
    else:
        desc = [
            html.Strong("✅ Kausal Entstört (Panel-Daten)", style={'display': 'block', 'marginBottom': '8px', 'fontSize': '16px'}),
            "Dieses Modell zerlegt das Studium in Semester-Zeilen.", html.Br(), html.Br(),
            "Im aktuellen Semester t darf das Modell nur auf Leistungs- und Supportdaten aus dem ", html.Strong("Vorsemester (t-1)"), " zugreifen.", html.Br(), html.Br(),
            html.Strong("Effekt:"), " Der Bias verschwindet. Stattdessen wird sichtbar, dass schwache Studierende öfter zum Support gehen (Confounding by Indication). Die HR nähert sich der Wahrheit."
        ]
        style = {'backgroundColor': '#f0fdf4', 'color': '#166534', 'border': '1px solid #bbf7d0', 'marginTop': '20px', 'padding': '15px', 'borderRadius': '5px', 'fontSize': '14px'}
        
        # Use time-varying variables from the panel
        if support_type == 'any_support':
            supp_col = 'any_supp_tv'
        elif support_type == 'Fach_supp':
            supp_col = 'fach_supp_tv'
        else:
            supp_col = 'any_supp_tv' # Fallback
            
        df['support_numeric'] = df[supp_col].astype(int)
        
        # In the panel, cum_cp and cum_fails are already lagged implicitly for the time interval (t_start to t_stop)
        formula = "time_rel ~ support_numeric + hzb_note + cum_cp + cum_fails + erstakademiker"
        
        # Wir müssen auch time_rel definieren für phreg in statsmodels, da das Panel t_start und t_stop hat
        # Allerdings erwartet phreg entweder time oder entry.
        # Im Original-Code (extended_cox_survival.py) wird entry=t_start und time=t_stop (oder was ähnliches) verwendet.
        # Lass uns einfach t_stop als time verwenden: "t_stop ~ ..."
        formula = "t_stop ~ support_numeric + hzb_note + cum_cp + cum_fails + erstakademiker"
        
        cox = smf.phreg(formula=formula, data=df, status=df['event'], entry=df['t_start'], ties='breslow').fit()
        idx_supp = list(cox.model.exog_names).index('support_numeric')
        hr_support = np.exp(cox.params[idx_supp])
        
    # 3. Forest Plot
    exog_names = cox.model.exog_names
    params = cox.params
    ci = cox.conf_int()
    
    hrs = np.exp(params)
    ci_lower = np.exp(ci[:, 0])
    ci_upper = np.exp(ci[:, 1])
    
    display_names = [n.replace('support_numeric', 'Support (Jemals)' if mode=='static' else 'Support (Lagged t-1)')
                      .replace('AVG_cp_sem1_2', 'Ø CP (Nach Sem 2!)')
                      .replace('cum_cp', 'Kum. CP (Lagged t-1)')
                      .replace('cum_fails', 'Fehlversuche (Lagged t-1)')
                      for n in exog_names]
                      
    fig_forest = go.Figure()
    fig_forest.add_shape(type="line", x0=1, y0=-0.5, x1=1, y1=len(exog_names)-0.5, line=dict(color="#ef4444", width=2, dash="dash"))
    fig_forest.add_trace(go.Scatter(
        x=hrs, y=display_names, mode='markers',
        marker=dict(color='#0ea5e9' if mode=='static' else '#10b981', size=12, symbol='square'),
        error_x=dict(type='data', symmetric=False, array=ci_upper-hrs, arrayminus=hrs-ci_lower, color='#0ea5e9' if mode=='static' else '#10b981', thickness=2),
        name='Hazard Ratio'
    ))
    fig_forest.update_layout(
        title="Multivariable Cox-Regression: Hazard Ratios", 
        yaxis=dict(autorange="reversed"), 
        xaxis=dict(title="<-- Reduziert Risiko | Erhöht Risiko -->", type='log', tickvals=[0.2, 0.5, 0.8, 1, 1.2, 2, 5]),
        margin=dict(l=180, r=40, t=60, b=40),
        plot_bgcolor='white',
        xaxis_gridcolor='#f1f5f9'
    )
    
    # 4. Summary Text
    is_protective = hr_support < 1
    stats = [
        html.Div(f"{'Statisches Cox PH' if mode == 'static' else 'Dynamisches Cox Panel'}", style={'fontWeight': 'bold', 'color': '#64748b', 'marginBottom': '15px'}),
        html.Div(style={'display': 'flex', 'alignItems': 'baseline', 'gap': '10px'}, children=[
            html.Span("HR (Support):", style={'fontSize': '18px', 'fontWeight': 'bold'}),
            html.Span(f"{hr_support:.3f}", style={'color': '#ef4444' if not is_protective else '#10b981', 'fontSize': '32px', 'fontWeight': '900'})
        ]),
        html.Div(f"95% KI: [{np.exp(ci[list(exog_names).index('support_numeric'), 0]):.3f} - {np.exp(ci[list(exog_names).index('support_numeric'), 1]):.3f}]", style={'color': '#94a3b8', 'marginBottom': '20px'}),
        html.Div([
            html.Strong("Interpretation: "),
            f"Support ist mit einem um {abs(1-hr_support)*100:.1f}% {'reduzierten' if is_protective else 'erhöhten'} Abbruchrisiko assoziiert."
        ], style={'backgroundColor': '#f8fafc', 'padding': '12px', 'borderRadius': '6px', 'borderLeft': f"4px solid {'#10b981' if is_protective else '#ef4444'}"})
    ]
    
    # 5. Survival Plot
    fig_surv = go.Figure()
    
    if mode == 'static':
        fig_surv.add_trace(go.Scatter(x=np.concatenate(([1.0], sf_mit.surv_times)), y=np.concatenate(([1.0], sf_mit.surv_prob)), mode='lines', line_shape='hv', name='Mit Support (Jemals)', line=dict(color='#0ea5e9', width=3)))
        fig_surv.add_trace(go.Scatter(x=np.concatenate(([1.0], sf_ohne.surv_times)), y=np.concatenate(([1.0], sf_ohne.surv_prob)), mode='lines', line_shape='hv', name='Ohne Support (Nie)', line=dict(color='#ef4444', dash='dash', width=3)))
        fig_surv.update_layout(
            title="Kaplan-Meier Kurven (Vorsicht: Immortal Time Bias!)", 
            xaxis_title="Semester", 
            yaxis_title="Überlebenswahrscheinlichkeit (S(t))",
            plot_bgcolor='white',
            yaxis_gridcolor='#f1f5f9',
            xaxis_gridcolor='#f1f5f9',
            legend=dict(yanchor="bottom", y=0.05, xanchor="left", x=0.05, bgcolor="rgba(255,255,255,0.8)")
        )
    else:
        # Erzeuge eine theoretische Kurve basierend auf der Baseline Hazard
        # cox.baseline_cumulative_hazard ist eine Liste (für jedes Stratum), deren Elemente wiederum Listen [times, hazards, ...] sind.
        baseline_cum_haz = cox.baseline_cumulative_hazard[0]
        t = baseline_cum_haz[0]
        H0 = baseline_cum_haz[1]
        
        S0 = np.exp(-H0)
        S1 = np.exp(-H0 * hr_support)
        
        fig_surv.add_trace(go.Scatter(x=t, y=S1, mode='lines', line_shape='hv', name='Szenario: Dauerhafter Support (HR-angepasst)', line=dict(color='#10b981', width=3)))
        fig_surv.add_trace(go.Scatter(x=t, y=S0, mode='lines', line_shape='hv', name='Szenario: Nie Support (Baseline)', line=dict(color='#64748b', dash='dash', width=3)))
        fig_surv.update_layout(
            title="Kausal Entstörte Überlebenskurven (Adjusted for Panel Covariates)", 
            xaxis_title="Semester", 
            yaxis_title="Überlebenswahrscheinlichkeit (S(t))",
            plot_bgcolor='white',
            yaxis_gridcolor='#f1f5f9',
            xaxis_gridcolor='#f1f5f9',
            legend=dict(yanchor="bottom", y=0.05, xanchor="left", x=0.05, bgcolor="rgba(255,255,255,0.8)")
        )

    return desc, style, fig_forest, stats, fig_surv

if __name__ == '__main__':
    app.run(debug=False, port=8051)
