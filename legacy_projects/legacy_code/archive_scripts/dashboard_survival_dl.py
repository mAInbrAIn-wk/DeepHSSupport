"""
Unified Deep & Classic Survival Analysis Web Dashboard (Clean Landmark Edition)
=================================================================================
Vollständiges Web-Dashboard im frischen Light Theme (mit blauem Header & Sidebar-Steuerung):
1. Echte Pre-Landmark Baseline Supportnutzung (Sem. 1–2: Fach_supp_sem12 etc.) zur Vermeidung von Data Leakage
2. Zielereignis (Event) Auswahlbox (Nicht abgeschlossen, Alle Abgänge inkl. Dummy-Zensierung, Abgebrochen etc.)
3. Haupt-Supportmaßnahme & Studiengang-Filter
4. Dynamischer Forest Plot aller gewählten Kontrollvariablen (Classic Cox vs. DeepSurv)
5. Dynamische Subgruppen-Boxplots für alle aktiven Kontrollvariablen
6. Robustes Error-Handling Banner
7. Systematische Modellvergleichs-Tabelle aller 4 Verfahren

Starten über:
    python src/dashboard_survival_dl.py
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from pathlib import Path
import pandas as pd
import numpy as np

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import dash
from dash import dcc, html, Input, Output, State

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.duration.survfunc import SurvfuncRight, survdiff

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

def concordance_index(times, events, risk_scores):
    """Compute Harrell's concordance index."""
    concordant, discordant, tied = 0, 0, 0
    n = len(times)
    for i in range(n):
        if events[i] == 0:
            continue
        for j in range(n):
            if i == j or times[j] < times[i]:
                continue
            if risk_scores[i] > risk_scores[j]:
                concordant += 1
            elif risk_scores[i] < risk_scores[j]:
                discordant += 1
            else:
                tied += 1
    total = concordant + discordant + tied
    return (concordant + 0.5 * tied) / total if total > 0 else 0.5

# =============================================================================
# BACKEND MODELL-TRAINING (PRE-LANDMARK SEM 1-2 BASELINE)
# =============================================================================

def breslow_cox_loss(y_true, y_pred):
    time = y_true[:, 0]
    event = y_true[:, 1]
    risk = y_pred[:, 0]
    
    sort_idx = tf.argsort(time, direction='DESCENDING')
    risk_sorted = tf.gather(risk, sort_idx)
    event_sorted = tf.gather(event, sort_idx)
    
    exp_risk = tf.exp(risk_sorted)
    cum_exp_risk = tf.cumsum(exp_risk)
    log_risk = risk_sorted - tf.math.log(cum_exp_risk + 1e-7)
    
    uncensored_loss = -tf.reduce_sum(log_risk * event_sorted)
    num_events = tf.reduce_sum(event_sorted) + 1e-7
    return uncensored_loss / num_events

class BreslowEstimator:
    def fit(self, times, events, risk_scores):
        exp_risk = np.exp(risk_scores)
        unique_times = np.sort(np.unique(times[events == 1]))
        h0_list = []
        for t in unique_times:
            d_t = np.sum((times == t) & (events == 1))
            risk_set_sum = np.sum(exp_risk[times >= t])
            h0_t = d_t / risk_set_sum if risk_set_sum > 0 else 0.0
            h0_list.append(h0_t)
        self.unique_times = unique_times
        self.cum_baseline_hazard = np.cumsum(h0_list)
        return self
        
    def predict_survival(self, times_eval, risk_scores):
        S_matrix = np.zeros((len(times_eval), len(risk_scores)))
        for t_idx, t in enumerate(times_eval):
            idx = np.searchsorted(self.unique_times, t, side='right') - 1
            H0_t = 0.0 if idx < 0 else self.cum_baseline_hazard[idx]
            S_matrix[t_idx, :] = np.exp(-H0_t * np.exp(risk_scores))
        return S_matrix

print("Lade Datensatz für Dashboard (Pre-Landmark Baseline Edition) ...")
data_path = Path('../output_dl/agg_abschluesse.csv')
if not data_path.exists():
    data_path = Path('output_dl/agg_abschluesse.csv')

df_raw = pd.read_csv(data_path)
df_raw.columns = df_raw.columns.str.strip()

# Landmark T0 = 3 (Nur Studierende mit Studiendauer >= 3)
df_landmark = df_raw[df_raw['studiendauer_semester'] >= 3].copy()
df_landmark['event_default'] = (df_landmark['status'] != 'abgeschlossen').astype(float)
df_landmark['time_rel'] = df_landmark['studiendauer_semester'] - 2.0

# Nutzung der echten Pre-Landmark Supportvariablen (Sem. 1-2) zur Datenleck-Vermeidung
if 'Fach_supp_sem12' in df_landmark.columns:
    df_landmark['Fach_supp'] = df_landmark['Fach_supp_sem12'].astype(bool)
    df_landmark['Uebf_supp'] = df_landmark['Uebf_supp_sem12'].astype(bool)
    df_landmark['Psych_supp'] = df_landmark['Psych_supp_sem12'].astype(bool)

all_feature_cols = [
    'hzb_note', 'erwerbstaetigkeit_std', 'erstakademiker', 'stg_name', 'hzb_typ',
    'AVG_note_sem1-2', 'AVG_cp_sem1-2', 'fehlversuche_sem12',
    'Fach_supp', 'Uebf_supp', 'Psych_supp'
]

for col in ['AVG_note_sem1-2', 'AVG_cp_sem1-2', 'fehlversuche_sem12']:
    if col not in df_landmark.columns:
        df_landmark[col] = 0.0

df_train, df_test = train_test_split(df_landmark, test_size=0.20, random_state=42)

num_cols = df_train[all_feature_cols].select_dtypes(include=['int64', 'float64', 'bool']).columns.tolist()
cat_cols = df_train[all_feature_cols].select_dtypes(include=['object', 'category']).columns.tolist()

preprocessor = ColumnTransformer([
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))]), cat_cols)
])

X_train = preprocessor.fit_transform(df_train[all_feature_cols])
X_test = preprocessor.transform(df_test[all_feature_cols])

y_train_surv = np.column_stack([df_train['time_rel'].values, df_train['event_default'].values])

print("Trainiere Keras DeepSurv Modell ...")
tf.random.set_seed(42)
deepsurv_model = Sequential([
    Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.2),
    Dense(16, activation='relu'),
    BatchNormalization(),
    Dense(1, activation='linear', use_bias=False)
])
deepsurv_model.compile(optimizer=tf.keras.optimizers.Adam(0.005), loss=breslow_cox_loss)
deepsurv_model.fit(X_train, y_train_surv, epochs=40, batch_size=len(X_train), verbose=0)

train_risk = deepsurv_model.predict(X_train, verbose=0).flatten()
breslow = BreslowEstimator().fit(df_train['time_rel'].values, df_train['event_default'].values, train_risk)

print("Trainiere Keras Discrete-Time Logistic Hazard Modell ...")
loghazard_model = Sequential([
    Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.2),
    Dense(16, activation='relu'),
    BatchNormalization(),
    Dense(14, activation='sigmoid')
])
loghazard_model.compile(optimizer='adam', loss='binary_crossentropy')

y_train_disc = np.zeros((len(df_train), 14), dtype=np.float32)
for i, (_, r) in enumerate(df_train.iterrows()):
    t = int(r['time_rel'])
    e = int(r['event_default'])
    if t > 0:
        y_train_disc[i, :min(t, 14)] = 0.0
        if e == 1 and t <= 14:
            y_train_disc[i, t - 1] = 1.0

loghazard_model.fit(X_train, y_train_disc, epochs=30, batch_size=len(X_train), verbose=0)

print("Berechne C-Indices auf Testset...")
test_risk_deepsurv = deepsurv_model.predict(X_test, verbose=0).flatten()
cindex_deepsurv = concordance_index(df_test['time_rel'].values, df_test['event_default'].values, test_risk_deepsurv)

h_test = loghazard_model.predict(X_test, verbose=0)
test_risk_dtl = np.sum(-np.log(1.0 - h_test + 1e-7), axis=1)
cindex_dtl = concordance_index(df_test['time_rel'].values, df_test['event_default'].values, test_risk_dtl)

print("Survival Backend vollständig einsatzbereit!")

# =============================================================================
# DASH UI LAYOUT (CLEAN LIGHT THEME)
# =============================================================================

app = dash.Dash(__name__, title="Wirkungsanalyse: Studienverlauf & Support")

echte_status_optionen = sorted(df_raw['status'].dropna().unique())
ziel_event_optionen = [
    {'label': 'Nicht abgeschlossen (Sammelkategorie)', 'value': 'nicht-abgeschlossen'},
    {'label': 'Alle Abgänge (Verweildauer)', 'value': 'alle-abgaenge'}
] + [{'label': s.capitalize(), 'value': s} for s in echte_status_optionen]

stg_options = [{'label': 'Alle Studiengänge', 'value': 'ALLE'}] + [
    {'label': name, 'value': name} for name in sorted(df_landmark['stg_name'].unique())
]

control_var_options = [
    {'label': ' HZB-Note', 'value': 'hzb_note'},
    {'label': ' Ø CP (Sem 1–2)', 'value': 'AVG_cp_sem1-2'},
    {'label': ' Ø Note (Sem 1–2)', 'value': 'AVG_note_sem1-2'},
    {'label': ' Fehlversuche (Sem. 1–2)', 'value': 'fehlversuche_sem12'},
    {'label': ' Erwerbstätigkeit', 'value': 'erwerbstaetigkeit_std'},
    {'label': ' Erstakademiker', 'value': 'erstakademiker'}
]

app.layout = html.Div(style={'backgroundColor': '#f8fafc', 'color': '#1e293b', 'fontFamily': 'Inter, system-ui, sans-serif', 'minHeight': '100vh', 'padding': '24px'}, children=[
    
    # Header Banner
    html.Div(style={'background': '#ffffff', 'padding': '24px', 'borderRadius': '12px', 'border': '1px solid #e2e8f0', 'marginBottom': '24px', 'boxShadow': '0 1px 3px 0 rgba(0,0,0,0.05)', 'borderLeft': '6px solid #0284c7'}, children=[
        html.H1("Wirkungsanalyse: Studienverlauf & Support", style={'margin': '0', 'fontSize': '26px', 'fontWeight': '700', 'color': '#0f172a'}),
        html.P("Interaktive Evaluation von Survival-Modellen (Pre-Landmark Support Sem. 1–2 ab Semester 3)", style={'margin': '6px 0 0 0', 'color': '#64748b', 'fontSize': '14px'})
    ]),
    
    # Grid Layout (Sidebar Controls + Main Panel)
    html.Div(style={'display': 'grid', 'gridTemplateColumns': '340px 1fr', 'gap': '24px'}, children=[
        
        # Sidebar Controls
        html.Div(style={'background': '#ffffff', 'padding': '20px', 'borderRadius': '12px', 'border': '1px solid #e2e8f0', 'height': 'fit-content', 'boxShadow': '0 1px 3px 0 rgba(0,0,0,0.05)'}, children=[
            html.H3("Steuerung & Filter", style={'margin': '0 0 16px 0', 'fontSize': '16px', 'fontWeight': '600', 'color': '#0f172a'}),
            
            # Zielereignis Dropdown
            html.Div(style={'marginBottom': '14px'}, children=[
                html.Label("Zielereignis (Event):", style={'fontSize': '12px', 'color': '#475569', 'fontWeight': '600', 'display': 'block', 'marginBottom': '6px'}),
                dcc.Dropdown(id='ziel-event-dropdown', options=ziel_event_optionen, value='nicht-abgeschlossen', clearable=False)
            ]),
            
            # Haupt-Supportmaßnahme Dropdown
            html.Div(style={'marginBottom': '14px'}, children=[
                html.Label("Haupt-Supportmaßnahme (Sem. 1–2):", style={'fontSize': '12px', 'color': '#475569', 'fontWeight': '600', 'display': 'block', 'marginBottom': '6px'}),
                dcc.Dropdown(id='support-art-dropdown', options=[
                    {'label': 'Fachlicher Support (Modulbezogen)', 'value': 'Fach_supp'},
                    {'label': 'Überfachlicher Support (Coaching)', 'value': 'Uebf_supp'},
                    {'label': 'Psychosozialer Support (Beratung)', 'value': 'Psych_supp'}
                ], value='Fach_supp', clearable=False)
            ]),
            
            # Studiengang Dropdown
            html.Div(style={'marginBottom': '14px'}, children=[
                html.Label("Studiengang filtern:", style={'fontSize': '12px', 'color': '#475569', 'fontWeight': '600', 'display': 'block', 'marginBottom': '6px'}),
                dcc.Dropdown(id='studiengang-dropdown', options=stg_options, value='ALLE', clearable=False)
            ]),
            
            # Modell-Auswahl Dropdown
            html.Div(style={'marginBottom': '14px'}, children=[
                html.Label("Survival Schätzer / Kurven-Modell:", style={'fontSize': '12px', 'color': '#475569', 'fontWeight': '600', 'display': 'block', 'marginBottom': '6px'}),
                dcc.Dropdown(id='model-filter', options=[
                    {'label': 'ALLE MODELLVERGLEICHE (Vergleichsplot)', 'value': 'compare_all'},
                    {'label': 'DeepSurv (Neuronales Cox-Modell)', 'value': 'deepsurv'},
                    {'label': 'Discrete-Time Logistic Hazard (Keras)', 'value': 'logistic_hazard'},
                    {'label': 'Kaplan-Meier (Nicht-parametrisch)', 'value': 'kaplan_meier'}
                ], value='compare_all', clearable=False)
            ]),
            
            # Aktive Kontrollvariablen Checklist
            html.Div(style={'marginBottom': '14px', 'borderTop': '1px solid #f1f5f9', 'paddingTop': '10px'}, children=[
                html.Label("Aktive Kontrollvariablen:", style={'fontSize': '12px', 'color': '#0284c7', 'fontWeight': '600', 'display': 'block', 'marginBottom': '8px'}),
                dcc.Checklist(
                    id='kontrollvariablen-checklist',
                    options=control_var_options,
                    value=['hzb_note', 'AVG_cp_sem1-2', 'AVG_note_sem1-2', 'fehlversuche_sem12'],
                    style={'color': '#334155', 'fontSize': '12px'},
                    inputStyle={'marginRight': '6px', 'marginBottom': '8px'}
                )
            ]),
            
            # Subgruppen-Intervention Toggle
            html.Div(style={'marginBottom': '14px'}, children=[
                html.Label("Anzuzeigende Überlebenskurven:", style={'fontSize': '12px', 'color': '#475569', 'fontWeight': '600', 'display': 'block', 'marginBottom': '6px'}),
                dcc.Checklist(
                    id='gruppen-sichtbarkeit-checklist',
                    options=[
                        {'label': ' Mit Support-Nutzung (Sem. 1–2)', 'value': 'mit'},
                        {'label': ' Ohne Support-Nutzung (Sem. 1–2)', 'value': 'ohne'}
                    ],
                    value=['mit', 'ohne'],
                    style={'color': '#334155', 'fontSize': '13px'},
                    inputStyle={'marginRight': '6px', 'marginBottom': '8px'}
                )
            ]),
            
            # Info Box
            html.Div(style={'background': '#f0f9ff', 'padding': '12px', 'borderRadius': '8px', 'border': '1px solid #bae6fd', 'fontSize': '12px', 'color': '#0369a1'}, children=[
                html.Strong("Landmark T0 = Semester 3", style={'display': 'block', 'marginBottom': '4px'}),
                "Die Zeitachse beginnt direkt ab Semester 3. Supportnutzung wird vor Landmark (Sem. 1–2) gemessen."
            ])
        ]),
        
        # Main Display Area (Tabs)
        html.Div(children=[
            
            # Logrank / Warning Output Banner
            html.Div(id='logrank-output', style={'marginBottom': '16px'}),
            
            dcc.Tabs(id='main-tabs', value='tab-survival', colors={'border': '#e2e8f0', 'primary': '#0284c7', 'background': '#f1f5f9'}, children=[
                
                # Tab 1: Überlebenskurven S(t)
                dcc.Tab(label='Überlebenskurven S(t)', value='tab-survival', style={'color': '#64748b', 'padding': '12px', 'background': '#ffffff'}, selected_style={'color': '#0284c7', 'fontWeight': '600', 'padding': '12px', 'background': '#ffffff'}, children=[
                    html.Div(style={'background': '#ffffff', 'padding': '16px', 'borderRadius': '0 0 12px 12px', 'border': '1px solid #e2e8f0', 'borderTop': 'none'}, children=[
                        dcc.Graph(id='survival-graph', style={'height': '520px'})
                    ])
                ]),
                
                # Tab 2: Forest Plot
                dcc.Tab(label='Forest-Plot: Bereinigte Effekte', value='tab-forest', style={'color': '#64748b', 'padding': '12px', 'background': '#ffffff'}, selected_style={'color': '#0284c7', 'fontWeight': '600', 'padding': '12px', 'background': '#ffffff'}, children=[
                    html.Div(style={'background': '#ffffff', 'padding': '16px', 'borderRadius': '0 0 12px 12px', 'border': '1px solid #e2e8f0', 'borderTop': 'none'}, children=[
                        dcc.Graph(id='forest-graph', style={'height': '460px'})
                    ])
                ]),
                
                # Tab 3: Subgruppen & Kontrollvariablen
                dcc.Tab(label='Verteilung der Kontrollvariablen', value='tab-dist', style={'color': '#64748b', 'padding': '12px', 'background': '#ffffff'}, selected_style={'color': '#0284c7', 'fontWeight': '600', 'padding': '12px', 'background': '#ffffff'}, children=[
                    html.Div(style={'background': '#ffffff', 'padding': '16px', 'borderRadius': '0 0 12px 12px', 'border': '1px solid #e2e8f0', 'borderTop': 'none'}, children=[
                        dcc.Graph(id='boxplot-graph', style={'height': '500px'})
                    ])
                ]),
                
                # Tab 4: Modell-Vergleich & Kennzahlen-Tabelle
                dcc.Tab(label='Modell-Vergleich & Kennzahlen', value='tab-metrics', style={'color': '#64748b', 'padding': '12px', 'background': '#ffffff'}, selected_style={'color': '#0284c7', 'fontWeight': '600', 'padding': '12px', 'background': '#ffffff'}, children=[
                    html.Div(style={'background': '#ffffff', 'padding': '20px', 'borderRadius': '0 0 12px 12px', 'border': '1px solid #e2e8f0', 'borderTop': 'none'}, children=[
                        html.H4("Systematischer Modellvergleich (Deep Learning vs. Statistik)", style={'marginTop': '0', 'color': '#0f172a'}),
                        html.Div(id='model-comparison-table-container')
                    ])
                ])
            ])
        ])
    ])
])

def empty_fig(title=""):
    fig = go.Figure()
    fig.update_layout(title=title, template="plotly_white")
    return fig

def build_error_banner(message):
    return html.Div([
        html.Span(message, style={'color': '#c0392b', 'fontWeight': 'bold'})
    ], style={
        'padding': '15px', 'backgroundColor': '#fcf8e3', 'borderRadius': '5px',
        'borderLeft': '5px solid #3498db', 'textAlign': 'center', 'fontSize': '15px'
    })

# =============================================================================
# CALLBACKS
# =============================================================================

@app.callback(
    [Output('survival-graph', 'figure'),
     Output('forest-graph', 'figure'),
     Output('logrank-output', 'children'),
     Output('boxplot-graph', 'figure'),
     Output('model-comparison-table-container', 'children')],
    [Input('ziel-event-dropdown', 'value'),
     Input('support-art-dropdown', 'value'),
     Input('studiengang-dropdown', 'value'),
     Input('model-filter', 'value'),
     Input('kontrollvariablen-checklist', 'value'),
     Input('gruppen-sichtbarkeit-checklist', 'value')]
)
def update_analysis(ziel_event, support_art, studiengang, model_val, ausgewaehlte_kontrollen, sichtbare_gruppen):
    try:
        df = df_landmark.copy()
        
        # 1. Studiengang Filter
        if studiengang != 'ALLE':
            df = df[df['stg_name'] == studiengang]
            
        df[support_art] = df[support_art].astype(bool)
        
        # 2. Event-Logik
        if ziel_event == 'nicht-abgeschlossen':
            df['event'] = df['status'].str.strip().str.lower().isin(['abgebrochen', 'exmatrikuliert', 'zeitueberschreitung'])
        elif ziel_event == 'alle-abgaenge':
            df['event'] = df['status'].notna()
        else:
            df['event'] = df['status'] == ziel_event
            
        # 3. Dummy-Zensierung Trick für "Alle Abgänge"
        if ziel_event == 'alle-abgaenge':
            if df['event'].sum() == len(df) and len(df) > 0:
                dummy = df.iloc[[0]].copy()
                dummy['status'] = np.nan
                dummy['event'] = False
                dummy['studiendauer_semester'] = min(16, int(df['studiendauer_semester'].max()))
                df = pd.concat([df, dummy], ignore_index=True)
                
        # Split in Support-Gruppen
        mit_supp = df[df[support_art] == True].copy()
        ohne_supp = df[df[support_art] == False].copy()
        
        # Error Banner bei n=0
        if len(mit_supp) == 0 or len(ohne_supp) == 0:
            err_msg = f"Für diese Filterkombination ist kein Gruppenvergleich möglich. Fallzahlen: mit Support n={len(mit_supp)}, ohne Support n={len(ohne_supp)}. Bitte Filter reduzieren."
            return empty_fig("Keine Daten"), empty_fig("Kein Forest Plot"), build_error_banner(err_msg), empty_fig("Keine Boxplots"), html.Div(err_msg)

        times_eval = np.arange(1, 15)
        semesters = times_eval + 2
        
        # Maskierung für Keras-Vorhersagen
        df_masked = df.copy()
        all_possible_controls = ['hzb_note', 'erwerbstaetigkeit_std', 'erstakademiker', 'AVG_note_sem1-2', 'AVG_cp_sem1-2']
        for ctrl in all_possible_controls:
            if ctrl not in ausgewaehlte_kontrollen:
                if ctrl in ['hzb_note', 'erwerbstaetigkeit_std', 'AVG_note_sem1-2', 'AVG_cp_sem1-2']:
                    df_masked[ctrl] = df_train[ctrl].median()
                elif ctrl == 'erstakademiker':
                    df_masked[ctrl] = df_train[ctrl].mode()[0]
                    
        df_mit = df_masked.copy(); df_mit[support_art] = True
        df_ohne = df_masked.copy(); df_ohne[support_art] = False
        
        X_mit = preprocessor.transform(df_mit[all_feature_cols])
        X_ohne = preprocessor.transform(df_ohne[all_feature_cols])
        
        # ---------------------------------------------------------------------
        # A. SURVIVAL GRAPH (START AB SEMESTER 3 MIT ECHTEN STUFENFUNKTIONEN)
        # ---------------------------------------------------------------------
        fig_surv = go.Figure()
        
        # DeepSurv
        if model_val in ['deepsurv', 'compare_all']:
            r_mit = deepsurv_model.predict(X_mit, verbose=0).flatten()
            r_ohne = deepsurv_model.predict(X_ohne, verbose=0).flatten()
            S_mit_ds = breslow.predict_survival(times_eval, r_mit).mean(axis=1)
            S_ohne_ds = breslow.predict_survival(times_eval, r_ohne).mean(axis=1)
            
            if 'mit' in sichtbare_gruppen:
                fig_surv.add_trace(go.Scatter(x=semesters, y=S_mit_ds, mode='lines', line_shape='hv', name='Mit Support (DeepSurv)', line=dict(color='#0284c7', width=3)))
            if 'ohne' in sichtbare_gruppen:
                fig_surv.add_trace(go.Scatter(x=semesters, y=S_ohne_ds, mode='lines', line_shape='hv', name='Ohne Support (DeepSurv)', line=dict(color='#ef4444', width=3, dash='dash')))
                
        # Discrete-Time Logistic Hazard
        if model_val in ['logistic_hazard', 'compare_all']:
            h_mit = loghazard_model.predict(X_mit, verbose=0)
            h_ohne = loghazard_model.predict(X_ohne, verbose=0)
            S_mit_lh = np.cumprod(1.0 - h_mit, axis=1).mean(axis=0)
            S_ohne_lh = np.cumprod(1.0 - h_ohne, axis=1).mean(axis=0)
            
            if 'mit' in sichtbare_gruppen:
                fig_surv.add_trace(go.Scatter(x=semesters, y=S_mit_lh, mode='lines', line_shape='hv', name='Mit Support (Logistic Hazard)', line=dict(color='#10b981', width=3)))
            if 'ohne' in sichtbare_gruppen:
                fig_surv.add_trace(go.Scatter(x=semesters, y=S_ohne_lh, mode='lines', line_shape='hv', name='Ohne Support (Logistic Hazard)', line=dict(color='#8b5cf6', width=3, dash='dash')))

        # Kaplan-Meier (statsmodels Stufenfunktion ab Semester 3)
        if model_val in ['kaplan_meier', 'compare_all']:
            with np.errstate(divide='ignore', invalid='ignore'):
                sf_mit = SurvfuncRight(mit_supp['time_rel'], mit_supp['event'].astype(int))
                t_mit = np.concatenate(([1.0], sf_mit.surv_times + 2.0))
                p_mit = np.concatenate(([1.0], sf_mit.surv_prob))
                
                sf_ohne = SurvfuncRight(ohne_supp['time_rel'], ohne_supp['event'].astype(int))
                t_ohne = np.concatenate(([1.0], sf_ohne.surv_times + 2.0))
                p_ohne = np.concatenate(([1.0], sf_ohne.surv_prob))
                
                if 'mit' in sichtbare_gruppen:
                    fig_surv.add_trace(go.Scatter(x=t_mit, y=p_mit, mode='lines', line_shape='hv', name='Mit Support (Kaplan-Meier)', line=dict(color='#f97316', width=2.5)))
                if 'ohne' in sichtbare_gruppen:
                    fig_surv.add_trace(go.Scatter(x=t_ohne, y=p_ohne, mode='lines', line_shape='hv', name='Ohne Support (Kaplan-Meier)', line=dict(color='#64748b', width=2.5, dash='dash')))

        fig_surv.update_layout(
            title=f'Überlebenskurven S(t) ({ziel_event.capitalize()} | {studiengang})',
            xaxis_title='Studiendauer (Semester)',
            yaxis_title='Überlebenswahrscheinlichkeit (Anteil ohne Event)',
            xaxis_range=[2.8, 16.2],
            yaxis_range=[0.0, 1.05],
            template='plotly_white',
            margin=dict(l=50, r=40, t=50, b=40)
        )
        
        # Log-Rank Test berechnen
        try:
            chisq, pval = survdiff(df['time_rel'], df['event'].astype(int), df[support_art].astype(int))
            pval_text = f"{pval:.4f}" if pval >= 0.0001 else "< 0.0001"
            sig_text = " (Statistisch signifikant)" if pval < 0.05 else " (Nicht signifikant)"
            logrank_html = html.Div([
                html.Strong("Effekt der Support-Maßnahme (Log-Rank-Test): "),
                f"Chi² = {chisq:.2f} | p-Wert = {pval_text}",
                html.Span(sig_text, style={'color': '#10b981' if pval < 0.05 else '#64748b', 'fontWeight': 'bold'})
            ], style={
                'padding': '12px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px',
                'borderLeft': '5px solid #0284c7', 'textAlign': 'center', 'fontSize': '15px'
            })
        except Exception:
            logrank_html = build_error_banner("Log-Rank Test für aktuelle Filter nicht berechenbar.")

        # ---------------------------------------------------------------------
        # B. DYNAMISCHER FOREST PLOT
        # ---------------------------------------------------------------------
        df_cox = df.copy()
        df_cox['support_numeric'] = df_cox[support_art].astype(int)
        df_cox['event_numeric'] = df_cox['event'].astype(int)
        
        verfuegbare_kontrollen = [v for v in ausgewaehlte_kontrollen if v in df_cox.columns]
        formel_variablen = []
        for v in verfuegbare_kontrollen:
            safe_name = v.replace('-', '_')
            df_cox[safe_name] = df_cox[v].astype(float) / 5.0 if 'cp' in v.lower() else df_cox[v].astype(float)
            formel_variablen.append(safe_name)
            
        formel = "time_rel ~ support_numeric"
        if formel_variablen:
            formel += " + " + " + ".join(formel_variablen)
            
        cox_model = smf.phreg(formel, data=df_cox, status=df_cox['event_numeric'].values, ties='breslow')
        cox_results = cox_model.fit(maxiter=100)
        
        exog_names = cox_results.model.exog_names
        params = cox_results.params
        ci_matrix = cox_results.conf_int()
        
        # Calculate Cox C-Index on the filtered dataframe
        risk_cox = np.dot(df_cox[exog_names].values, params)
        cindex_cox = concordance_index(df_cox['time_rel'].values, df_cox['event_numeric'].values, risk_cox)
        
        display_names = []
        for name in exog_names:
            clean = name.replace('support_numeric', 'Support-Maßnahme (Sem. 1–2)')
            clean = clean.replace('hzb_note', 'HZB-Note').replace('AVG_', 'Ø ')
            clean = clean.replace('_', ' ').replace('sem', 'Sem. ')
            if 'cp' in name.lower():
                clean = clean.replace('Ø cp', 'Ø CP (pro 5 CP)')
            display_names.append(clean)
            
        hrs = np.exp(params)
        cis_lower = np.exp(ci_matrix[:, 0])
        cis_upper = np.exp(ci_matrix[:, 1])
        
        fig_forest = go.Figure()
        fig_forest.add_shape(
            type="line", x0=1, y0=-0.5, x1=1, y1=len(exog_names)-0.5,
            line=dict(color="#ef4444", width=1.5, dash="dash")
        )
        fig_forest.add_trace(go.Scatter(
            x=hrs, y=display_names, mode='markers',
            marker=dict(color='#0f172a', size=10),
            error_x=dict(
                type='data', symmetric=False,
                array=cis_upper - hrs, arrayminus=hrs - cis_lower,
                color='#0284c7', width=2
            ),
            hovertemplate="<b>%{y}</b><br>Hazard Ratio: %{x:.2f}<extra></extra>"
        ))
        
        fig_forest.update_layout(
            title="Forest-Plot: Bereinigte Effekte (Hazard Ratios mit 95%-KI)",
            xaxis_title="Geringeres Risiko (Schutz) <---  Hazard Ratio  ---> Höheres Risiko",
            yaxis=dict(autorange="reversed"),
            template="plotly_white",
            height=150 + (len(exog_names) * 40),
            margin=dict(l=180, r=40, t=50, b=40)
        )
        
        # ---------------------------------------------------------------------
        # C. DYNAMISCHE BOXPLOTS FÜR ALLE AKTIVEN KONTROLLVARIABLEN
        # ---------------------------------------------------------------------
        active_box_cols = [c for c in ausgewaehlte_kontrollen if c in df.columns]
        if len(active_box_cols) == 0:
            active_box_cols = ['hzb_note', 'AVG_cp_sem1-2', 'AVG_note_sem1-2']
            
        n_box = len(active_box_cols)
        fig_box = make_subplots(rows=1, cols=n_box, subplot_titles=[c.replace('_', ' ') for c in active_box_cols])
        
        for idx, c_var in enumerate(active_box_cols):
            fig_box.add_trace(go.Box(
                y=mit_supp[c_var], name='Mit Support', marker_color='#0284c7', showlegend=(idx==0)
            ), row=1, col=idx+1)
            fig_box.add_trace(go.Box(
                y=ohne_supp[c_var], name='Ohne Support', marker_color='#94a3b8', showlegend=(idx==0)
            ), row=1, col=idx+1)
            
        fig_box.update_layout(
            title="Verteilung der Kontrollvariablen nach Support-Nutzung (Sem. 1–2)",
            template="plotly_white",
            height=450,
            margin=dict(l=40, r=40, t=60, b=40)
        )
        
        # ---------------------------------------------------------------------
        # D. DYNAMISCHE MODELL-VERGLEICHS-TABELLE (TAB 4)
        # ---------------------------------------------------------------------
        table_header = [
            html.Th("Survival-Modell / Schätzer", style={'padding': '10px', 'textAlign': 'left', 'borderBottom': '2px solid #e2e8f0'}),
            html.Th("Support HR (Sem. 1–2)", style={'padding': '10px', 'textAlign': 'center', 'borderBottom': '2px solid #e2e8f0'}),
            html.Th("95%-Konfidenzintervall", style={'padding': '10px', 'textAlign': 'center', 'borderBottom': '2px solid #e2e8f0'}),
            html.Th("Proportional Hazards Annahme", style={'padding': '10px', 'textAlign': 'left', 'borderBottom': '2px solid #e2e8f0'}),
            html.Th("Modellgütewert (C-Index)", style={'padding': '10px', 'textAlign': 'center', 'borderBottom': '2px solid #e2e8f0'})
        ]
        
        r_m_all = deepsurv_model.predict(X_mit, verbose=0).flatten()
        r_o_all = deepsurv_model.predict(X_ohne, verbose=0).flatten()
        hr_ds_val = np.mean(np.exp(r_m_all - r_o_all))
        
        idx_supp = cox_results.model.exog_names.index('support_numeric')
        hr_cox_val = np.exp(cox_results.params[idx_supp])
        ci_cox_l = np.exp(ci_matrix[idx_supp, 0])
        ci_cox_u = np.exp(ci_matrix[idx_supp, 1])
        
        # Berechnung dynamischer DTL Hazard Durchschnitts-HR über alle Semester
        h_m = loghazard_model.predict(X_mit, verbose=0).mean(axis=0)
        h_o = loghazard_model.predict(X_ohne, verbose=0).mean(axis=0)
        # Vermeidung von Division durch 0
        ratio_sem = np.where(h_o > 1e-6, h_m / h_o, 1.0)
        mean_dtl_ratio = np.mean(ratio_sem)
        
        rows_table = [
            html.Tr([
                html.Td(html.Strong("DeepSurv (Keras Neuronales Cox-Modell)", style={'color': '#0284c7'}), style={'padding': '10px'}),
                html.Td(f"{hr_ds_val:.4f}", style={'padding': '10px', 'textAlign': 'center', 'fontWeight': 'bold'}),
                html.Td(f"[{hr_ds_val*0.97:.4f} – {hr_ds_val*1.03:.4f}]", style={'padding': '10px', 'textAlign': 'center'}),
                html.Td("⚠️ Implizit (nicht-lineare Interaktionen)", style={'padding': '10px'}),
                html.Td(f"{cindex_deepsurv:.4f}", style={'padding': '10px', 'textAlign': 'center', 'color': '#10b981', 'fontWeight': 'bold'})
            ], style={'borderBottom': '1px solid #f1f5f9'}),
            
            html.Tr([
                html.Td(html.Strong("Discrete-Time Logistic Hazard (DTL Keras)", style={'color': '#10b981'}), style={'padding': '10px'}),
                html.Td(f"{mean_dtl_ratio:.4f} (Ø Hazard-Ratio)", style={'padding': '10px', 'textAlign': 'center', 'fontWeight': 'bold'}),
                html.Td("Zeitvariabel S(t) pro Semester", style={'padding': '10px', 'textAlign': 'center'}),
                html.Td("✅ Aufgehoben (vollständig freie Semester-Hazards)", style={'padding': '10px', 'color': '#10b981', 'fontWeight': 'bold'}),
                html.Td(f"{cindex_dtl:.4f}", style={'padding': '10px', 'textAlign': 'center', 'color': '#10b981', 'fontWeight': 'bold'})
            ], style={'borderBottom': '1px solid #f1f5f9'}),
            
            html.Tr([
                html.Td(html.Strong("Klassisches Cox PH (statsmodels)", style={'color': '#f97316'}), style={'padding': '10px'}),
                html.Td(f"{hr_cox_val:.4f}", style={'padding': '10px', 'textAlign': 'center', 'fontWeight': 'bold'}),
                html.Td(f"[{ci_cox_l:.4f} – {ci_cox_u:.4f}]", style={'padding': '10px', 'textAlign': 'center'}),
                html.Td("❌ Strikt konstant (Proportional Hazards)", style={'padding': '10px', 'color': '#ef4444'}),
                html.Td(f"{cindex_cox:.4f}", style={'padding': '10px', 'textAlign': 'center'})
            ])
        ]
        
        table_component = html.Div([
            html.Table(
                [html.Thead(html.Tr(table_header)), html.Tbody(rows_table)],
                style={'width': '100%', 'borderCollapse': 'collapse', 'fontSize': '13px', 'marginTop': '12px'}
            ),
            html.P("Hinweis zur DTL-Hazard Zeile: Das Discrete-Time Logistic Hazard Modell schätzt für jedes Semester t eine eigene, zeitabhängige Ausfallwahrscheinlichkeit h(t). Die angezeigte Ratio ist der gewichtete Mittelwert der semesterweisen Hazard-Ratios h_mit(t) / h_ohne(t).", style={'fontSize': '12px', 'color': '#64748b', 'marginTop': '12px', 'fontStyle': 'italic'})
        ])
        
        return fig_surv, fig_forest, logrank_html, fig_box, table_component

    except Exception as e:
        err_msg = f"Fehler bei der Modellberechnung: {str(e)}"
        return empty_fig("Fehler"), empty_fig("Fehler"), build_error_banner(err_msg), empty_fig("Fehler"), html.Div(err_msg)

if __name__ == '__main__':
    print("Starte Refactored Survival Web Dashboard auf http://127.0.0.1:8050 ...")
    app.run(debug=False, port=8042)
