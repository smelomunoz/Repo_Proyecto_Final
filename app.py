# app.py
import json
import os

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import plotly.express as px
import plotly.graph_objects as go

from dash import Dash, dcc, html, Input, Output

# =====================================================
# CONFIGURACIÓN DE RUTAS (AJUSTA AQUÍ SI ES NECESARIO)
# =====================================================

# CAMBIO: dejo todas las rutas juntas para que,
# si algo cambia, solo toques este bloque.
DF_REG_PATH = "df_cleaned.csv"  # datos base modelos 1 y 2
DF_CLF_PATH = "Repo_Proyecto_Final/Modelo_Clasificacion/df_modelo3_clasificacion.csv"

ART_DIR_LIN = "Repo_Proyecto_Final/Modelo_Regresion/artifacts_linreg_model1"
ART_DIR_REG = "Repo_Proyecto_Final/Modelo_Redes_Neuronales/artifacts_reg"
ART_DIR_CLF = "Repo_Proyecto_Final/Modelo_Clasificacion/artifacts_clf"


# =====================================================
# 1. CARGA DE DATAFRAMES Y CREACIÓN DE ID COMÚN
# =====================================================

# CAMBIO 1: ahora hay DOS dataframes y les creamos el MISMO prop_id
df_reg = pd.read_csv(DF_REG_PATH)       # usado por regresión lineal y redes
df_clf = pd.read_csv(DF_CLF_PATH)       # usado por clasificación y segmentos

# Asegurar que tengan el mismo número de filas y estén alineados
df_reg = df_reg.reset_index(drop=True)
df_clf = df_clf.reset_index(drop=True)

assert len(df_reg) == len(df_clf), (
    "df_cleaned y df_modelo3_clasificacion no tienen el mismo número de filas. "
    "Revisa que vengan del mismo dataset sin filtrados distintos."
)

# Crear ID común
prop_id = np.arange(len(df_reg))
df_reg["prop_id"] = prop_id
df_clf["prop_id"] = prop_id
id_col = "prop_id"

# Tipos básicos
df_reg["price"] = df_reg["price"].astype(float)
df_reg["accommodates"] = df_reg["accommodates"].astype(int)

df_clf["price"] = df_clf["price"].astype(float)
df_clf["accommodates"] = df_clf["accommodates"].astype(int)

# Si faltara price_per_guest en df_clf, lo creamos
if "price_per_guest" not in df_clf.columns:
    df_clf["price_per_guest"] = df_clf["price"] / df_clf["accommodates"]


# =====================================================
# 2. MODELO 1 – REGRESIÓN LINEAL (importancia de variables)
# =====================================================

with open(os.path.join(ART_DIR_LIN, "features_modelo1.json"), "r") as f:
    feature_info_lin = json.load(f)

feature_cols_lin = feature_info_lin["feature_cols"]

lin_reg = joblib.load(
    os.path.join(ART_DIR_LIN, "modelo1_regresion_lineal.pkl")
)

coef_df = pd.DataFrame({
    "feature": feature_cols_lin,
    "coef": lin_reg.coef_
})
coef_df["abs_coef"] = coef_df["coef"].abs()
coef_df_top = (
    coef_df.sort_values("abs_coef", ascending=False)
           .head(15)
)


# =====================================================
# 3. MODELO 2 – RED DE REGRESIÓN (rango de precio)
# =====================================================

modelo_reg = tf.keras.models.load_model(
    os.path.join(ART_DIR_REG, "modelo_regresion_final.keras")
)

with open(os.path.join(ART_DIR_REG, "config_reg.json"), "r") as f:
    config_reg = json.load(f)

feature_cols_nn = config_reg["feature_cols_nn"]
sigma_log_def = float(config_reg["sigma_log_def"])
k_default = float(config_reg["k_default"])

def recomendar_rango_una_propiedad(row, k=None):
    """Devuelve (precio_central, min, max) usando el modelo de redes."""
    if k is None:
        k = k_default

    X = row[feature_cols_nn].astype("float32").values.reshape(1, -1)
    log_price_pred = modelo_reg.predict(X, verbose=0)[0, 0]

    price_central = float(np.exp(log_price_pred))
    price_min = float(np.exp(log_price_pred - k * sigma_log_def))
    price_max = float(np.exp(log_price_pred + k * sigma_log_def))

    return price_central, price_min, price_max


# =====================================================
# 4. MODELO 3 – CLASIFICACIÓN (propiedad recomendada / no)
# =====================================================

modelo_clf = tf.keras.models.load_model(
    os.path.join(ART_DIR_CLF, "modelo_clasificacion_final.keras")
)

scaler_clf = joblib.load(
    os.path.join(ART_DIR_CLF, "scaler_clf.joblib")
)

with open(os.path.join(ART_DIR_CLF, "features_clf.json"), "r") as f:
    config_clf = json.load(f)

feature_cols_clf = config_clf["cols_features"]

# Umbral desde metrics_clf.json (si existe)
try:
    with open(os.path.join(ART_DIR_CLF, "metrics_clf.json"), "r") as f:
        metrics_clf = json.load(f)
    threshold_clf = metrics_clf.get("best_threshold", 0.5)
except FileNotFoundError:
    threshold_clf = 0.5

def predecir_recommended(df_props):
    """Devuelve probabilidad y etiqueta recomendada / no recomendada."""
    X = df_props[feature_cols_clf].values
    X_scaled = scaler_clf.transform(X)
    probs = modelo_clf.predict(X_scaled, verbose=0).flatten()
    labels = (probs >= threshold_clf).astype(int)
    return probs, labels

# CAMBIO: las probabilidades y etiquetas viven en df_clf
df_clf["prob_recommended"], df_clf["recommended_pred"] = predecir_recommended(df_clf)


# =====================================================
# 5. FUNCIONES DE VISUALIZACIÓN
# =====================================================

def fig_importancias():
    """Visualización 1 – importancia de variables (Modelo 1)."""
    fig = px.bar(
        coef_df_top.sort_values("coef"),
        x="coef",
        y="feature",
        orientation="h",
        title="Variables que más explican el log-precio (Modelo 1: regresión lineal)",
        labels={"coef": "Coeficiente (log(price))", "feature": "Variable"}
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=40, b=10),
        yaxis=dict(tickfont=dict(size=10))
    )
    return fig


def make_fig_rango_precio(prop_id_value):
    """
    Visualización 2 – Comparación simple:
    promedio del segmento vs precio recomendado (modelo) vs precio actual.
    NO modifica ningún código de los modelos, solo usa sus resultados.
    """
    prop_id_value = int(prop_id_value)

    # Fila en df_reg para usar el modelo de redes (precio, features, etc.)
    row_reg = df_reg.loc[df_reg[id_col] == prop_id_value].iloc[0]
    # Fila en df_clf para identificar segmento
    row_clf = df_clf.loc[df_clf[id_col] == prop_id_value].iloc[0]

    # Segmento: mismo borough y tipo de habitación (si existen)
    zona_col = "borough_seg"
    room_col = "room_type_seg"

    if zona_col in df_clf.columns and room_col in df_clf.columns:
        mask_seg = (
            (df_clf[zona_col] == row_clf[zona_col]) &
            (df_clf[room_col] == row_clf[room_col])
        )
        titulo_segmento = f"{row_clf[zona_col]} – {row_clf[room_col]}"
    else:
        mask_seg = np.ones(len(df_clf), dtype=bool)
        titulo_segmento = "Todo el dataset"

    df_seg = df_clf[mask_seg].copy()
    if df_seg.empty:
        df_seg = df_clf.copy()
        titulo_segmento = "Todo el dataset"

    # Promedio de precio en el segmento (solo datos, sin modelo)
    seg_mean_price = float(df_seg["price"].mean())

    # Precio recomendado por el modelo de redes (llamando a la función que ya tienes)
    price_central, price_min, price_max = recomendar_rango_una_propiedad(row_reg)
    price_real = float(row_reg["price"])

    # Gráfico de barras con 3 barras: promedio segmento, recomendado, actual
    etiquetas = [
        "Promedio segmento",
        "Precio recomendado (modelo)",
        "Precio actual"
    ]
    valores = [seg_mean_price, price_central, price_real]

    fig = px.bar(
        x=etiquetas,
        y=valores,
        labels={"x": "", "y": "Precio por noche (USD)"},
        title=f"Precio de la propiedad vs segmento y modelo (segmento: {titulo_segmento})"
    )

    fig.update_layout(
        margin=dict(l=10, r=10, t=40, b=10)
    )

    # Texto explicativo para debajo de la gráfica
    diff_model_abs = price_real - price_central
    diff_seg_abs = price_real - seg_mean_price

    sentido_model = "por encima" if diff_model_abs > 0 else "por debajo" if diff_model_abs < 0 else "igual a"
    sentido_seg = "por encima" if diff_seg_abs > 0 else "por debajo" if diff_seg_abs < 0 else "igual a"

    texto = (
        f"En el segmento {titulo_segmento}, el precio promedio observado es "
        f"{seg_mean_price:,.0f} USD por noche. "
        f"El Modelo 2 (red de regresión) recomienda aproximadamente "
        f"{price_central:,.0f} USD por noche. "
        f"La propiedad seleccionada cobra {price_real:,.0f} USD, es decir, está "
        f"{abs(diff_model_abs):,.0f} USD {sentido_model} del valor recomendado "
        f"por el modelo y {abs(diff_seg_abs):,.0f} USD {sentido_seg} del promedio "
        f"observado en su segmento."
    )

    return fig, texto


def make_fig_posicionamiento(prop_id_value):
    """
    Visualización 3 – Posicionamiento de la propiedad en el segmento
    según lo que ofrece (amenities_count), lo que cobra (price_per_guest)
    y la etiqueta del modelo (Recomendada / No recomendada).
    """
    prop_id_value = int(prop_id_value)

    # Fila seleccionada en df_clf
    row_clf = df_clf.loc[df_clf[id_col] == prop_id_value].iloc[0]

    zona_col = "borough_seg"
    room_col = "room_type_seg"

    # Definimos el segmento: mismo borough y tipo de habitación
    if zona_col in df_clf.columns and room_col in df_clf.columns:
        mask_seg = (
            (df_clf[zona_col] == row_clf[zona_col]) &
            (df_clf[room_col] == row_clf[room_col])
        )
        segmento_label = f"{row_clf[zona_col]} – {row_clf[room_col]}"
    else:
        mask_seg = np.ones(len(df_clf), dtype=bool)
        segmento_label = "Todo el dataset"

    df_seg = df_clf[mask_seg].copy()
    if df_seg.empty:
        df_seg = df_clf.copy()
        segmento_label = "Todo el dataset"

    # Columna categórica para colorear: Recomendada / No recomendada
    df_seg = df_seg.copy()
    df_seg["Etiqueta modelo"] = np.where(
        df_seg["recommended_pred"] == 1,
        "Recomendada",
        "No recomendada"
    )

    # Scatter: amenities vs price_per_guest, coloreado por etiqueta
    fig = px.scatter(
        df_seg,
        x="amenities_count",
        y="price_per_guest",
        color="Etiqueta modelo",
        labels={
            "amenities_count": "Número de amenidades",
            "price_per_guest": "Precio por huésped (USD)",
            "Etiqueta modelo": "Clasificación del modelo"
        },
        title=f"Posicionamiento en el segmento: {segmento_label}",
        hover_data=["prop_id", "price", "accommodates", "prob_recommended"]
    )

    # Estrella para la propiedad seleccionada
    fig.add_trace(go.Scatter(
        x=[row_clf["amenities_count"]],
        y=[row_clf["price_per_guest"]],
        mode="markers+text",
        name="Propiedad seleccionada",
        text=[f"Prop {prop_id_value}"],
        textposition="top center",
        marker=dict(size=16, symbol="star", line=dict(width=2))
    ))

    fig.update_layout(
        margin=dict(l=10, r=10, t=40, b=10)
    )

    # Texto explicativo
    prob = float(row_clf["prob_recommended"])
    etiqueta = "RECOMENDADA" if row_clf["recommended_pred"] == 1 else "NO RECOMENDADA"

    # Promedio de precio por huésped entre las recomendadas del segmento
    seg_mean_ppg = df_seg["price_per_guest"].mean()
    seg_mean_ppg_rec = df_seg.loc[df_seg["recommended_pred"] == 1, "price_per_guest"].mean()

    texto = (
        f"En el segmento {segmento_label}, el precio por huésped promedio es "
        f"{seg_mean_ppg:,.2f} USD, y entre las propiedades que el modelo clasifica "
        f"como recomendadas es de aproximadamente {seg_mean_ppg_rec:,.2f} USD. "
        f"La propiedad seleccionada tiene un precio por huésped de "
        f"{row_clf['price_per_guest']:,.2f} USD y una probabilidad de recomendación "
        f"estimada de {prob:.2%}, por lo que el Modelo 3 la clasifica como {etiqueta}."
    )

    return fig, texto


# =====================================================
# 6. CONSTRUCCIÓN DEL DASH
# =====================================================

app = Dash(__name__)

# CAMBIO: el dropdown se arma con df_clf porque tiene info descriptiva
muestra_dropdown = df_clf[["prop_id", "borough_seg", "room_type_seg",
                           "accommodates", "price"]].copy()

dropdown_options = [
    {
        "label": (
            f"{row['borough_seg']} | {row['room_type_seg']} | "
            f"{int(row['accommodates'])} huéspedes | {row['price']:.0f} USD"
        ),
        "value": int(row["prop_id"])
    }
    for _, row in muestra_dropdown.iterrows()
]

app.layout = html.Div(
    [
        html.H1("Tablero Airbnb NYC – Modelos predictivos",
                style={"textAlign": "center"}),

        html.H2("1. Variables que explican el precio por noche"),
        html.P(
            "Visualización basada en el Modelo 1 (regresión lineal) "
            "mostrando las 15 variables que más influyen en el log-precio."
        ),
        dcc.Graph(
            id="grafico_importancias",
            figure=fig_importancias()
        ),

        html.Hr(),

        html.H2("2 y 3. Análisis de una propiedad específica"),
        html.P(
            "Selecciona una propiedad para ver su rango de precio razonable "
            "(Modelo 2) y su posicionamiento / etiqueta recomendada "
            "(Modelo 3)."
        ),

        dcc.Dropdown(
            id="dropdown_propiedad",
            options=dropdown_options,
            value=int(muestra_dropdown.iloc[0]["prop_id"]),
            placeholder="Selecciona una propiedad...",
            style={"width": "70%"}
        ),

        html.Br(),

        html.H3("2. Distribución de precios y rango sugerido (Modelo 2: red de regresión)"),
        dcc.Graph(id="grafico_rango_precio"),
        html.Div(id="texto_rango_precio"),

        html.Hr(),

        html.H3("3. Posicionamiento y etiqueta recomendada (Modelo 3: clasificación)"),
        dcc.Graph(id="grafico_posicionamiento"),
        html.Div(id="texto_posicionamiento"),
    ],
    style={"margin": "20px"}
)


# =====================================================
# 7. CALLBACKS
# =====================================================

@app.callback(
    Output("grafico_rango_precio", "figure"),
    Output("texto_rango_precio", "children"),
    Input("dropdown_propiedad", "value")
)
def actualizar_rango_precio(prop_id_value):
    fig, texto = make_fig_rango_precio(prop_id_value)
    return fig, texto


@app.callback(
    Output("grafico_posicionamiento", "figure"),
    Output("texto_posicionamiento", "children"),
    Input("dropdown_propiedad", "value")
)
def actualizar_posicionamiento(prop_id_value):
    fig, texto = make_fig_posicionamiento(prop_id_value)
    return fig, texto


# =====================================================
# 8. PUNTO DE ENTRADA
# =====================================================

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)