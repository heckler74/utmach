import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


st.set_page_config(page_title="Ecuador Export Dashboard", page_icon="🌍", layout="wide")


st.markdown(
    """
    <style>
    .stApp {
        background: #ffffff;
        color: #1f2937;
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
    }
    [data-testid="stSidebar"] {
        background: #f8fafc;
        border-right: 1px solid #e5e7eb;
    }
    div[data-testid="stMetricValue"] {
        color: #0f172a;
        font-weight: 700;
    }
    .css-1d391kg, .css-1v0mbdj {
        background: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


DATA_PATH = Path("ecuadors-exports-to-world-in-2025-by-importer.xlsx")


@st.cache_data
def load_data():
    """Carga y normaliza el archivo Excel."""
    df = pd.read_excel(DATA_PATH)

    df.columns = [
        str(col).strip().replace("\xa0", " ").replace("  ", " ").strip()
        for col in df.columns
    ]

    rename_map = {
        "reporterCd": "reporterCd",
        "Importador": "Importador",
        "partnerCd": "partnerCd",
        "partnerLabel": "Producto",
        "productCd": "productCd",
        "productLabel": "productLabel",
        "Valor exportado ($)": "Valor exportado ($)",
        "Balanza comercial ($)": "Balanza comercial ($)",
        "Cantidad exportada": "Cantidad exportada",
        "Valor unitario": "Valor unitario",
        "Participación en las exportaciones de Ecuador (%)": "Participación en las exportaciones de Ecuador (%)",
        "Participación de Ecuador en las importaciones de socios (%)": "Participación de Ecuador en las importaciones de socios (%)",
        "Cuota del importador en las importaciones mundiales (%)": "Cuota del importador en las importaciones mundiales (%)",
        "Clasificación del importador en importaciones mundiales": "Clasificación del importador en importaciones mundiales",
        "Crecimiento anual de las exportaciones (2021-2025) (%)": "Crecimiento anual de las exportaciones (2021-2025) (%)",
        "Crecimiento anual de las exportaciones (2024-2025) (%)": "Crecimiento anual de las exportaciones (2024-2025) (%)",
        "Crecimiento anual del total de importaciones de socios (2021-2025) (%)": "Crecimiento anual del total de importaciones de socios (2021-2025) (%)",
        "Crecimiento anual en cantidad (2021-2025)": "Crecimiento anual en cantidad (2021-2025)",
    }
    df = df.rename(columns=rename_map)

    numeric_cols = [
        "Valor exportado ($)",
        "Balanza comercial ($)",
        "Cantidad exportada",
        "Valor unitario",
        "Participación en las exportaciones de Ecuador (%)",
        "Participación de Ecuador en las importaciones de socios (%)",
        "Cuota del importador en las importaciones mundiales (%)",
        "Clasificación del importador en importaciones mundiales",
        "Crecimiento anual de las exportaciones (2021-2025) (%)",
        "Crecimiento anual de las exportaciones (2024-2025) (%)",
        "Crecimiento anual del total de importaciones de socios (2021-2025) (%)",
        "Crecimiento anual en cantidad (2021-2025)",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Importador"] = df["Importador"].astype(str).str.strip()
    df["Producto"] = df["Producto"].astype(str).str.strip()

    df = df.dropna(subset=["Importador", "Producto", "Valor exportado ($)"]).reset_index(drop=True)
    return df


@st.cache_data
def get_dataset_overview(df: pd.DataFrame):
    """Genera resumen técnico del dataset para el panel descriptivo."""
    overview = {
        "filas": int(df.shape[0]),
        "columnas": int(df.shape[1]),
        "importadores_unicos": int(df["Importador"].nunique()),
        "productos_unicos": int(df["Producto"].nunique()),
        "valor_total": float(df["Valor exportado ($)"].sum()),
        "valor_promedio": float(df["Valor exportado ($)"].mean()),
        "cantidad_total": float(df["Cantidad exportada"].sum()),
        "missing_values": int(df.isna().sum().sum()),
    }
    return overview


@st.cache_data
def detect_outliers(df: pd.DataFrame, metric: str = "Valor exportado ($)"):
    q1 = df[metric].quantile(0.25)
    q3 = df[metric].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = df[(df[metric] < lower) | (df[metric] > upper)].copy()
    return q1, q3, iqr, lower, upper, outliers


@st.cache_resource
def train_rf_model(df: pd.DataFrame):
    """Entrena un Random Forest Regressor para predecir el valor exportado."""
    feature_cols = [
        "Cantidad exportada",
        "Valor unitario",
        "Participación en las exportaciones de Ecuador (%)",
        "Participación de Ecuador en las importaciones de socios (%)",
        "Cuota del importador en las importaciones mundiales (%)",
        "Clasificación del importador en importaciones mundiales",
        "Crecimiento anual de las exportaciones (2024-2025) (%)",
        "Crecimiento anual del total de importaciones de socios (2021-2025) (%)",
        "Crecimiento anual en cantidad (2021-2025)",
    ]

    model_df = df.dropna(subset=feature_cols + ["Valor exportado ($)"]).copy()
    model_df = pd.get_dummies(model_df, columns=["Importador", "Producto"], drop_first=False)

    X = model_df.drop(columns=["Valor exportado ($)"])
    y = model_df["Valor exportado ($)"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    model = RandomForestRegressor(
        n_estimators=250,
        max_depth=10,
        random_state=42,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    metrics = {
        "mae": mean_absolute_error(y_test, preds),
        "r2": r2_score(y_test, preds),
    }

    return model, metrics, list(X.columns)


def format_currency(value):
    return f"${value:,.0f}"


def build_descriptive_cards(filtered_df: pd.DataFrame):
    valor_total = filtered_df["Valor exportado ($)"].sum()
    valor_promedio = filtered_df["Valor exportado ($)"].mean()
    cantidad_total = filtered_df["Cantidad exportada"].sum()
    crecimiento_promedio = filtered_df["Crecimiento anual de las exportaciones (2024-2025) (%)"].mean()
    mercado_lider = filtered_df.groupby("Importador")["Valor exportado ($)"].sum().sort_values(ascending=False).index[0]

    kpis = [
        ("Valor exportado total", format_currency(valor_total)),
        ("Promedio por observación", format_currency(valor_promedio)),
        ("Cantidad exportada total", f"{cantidad_total:,.0f}"),
        ("Crecimiento 2024-2025 promedio", f"{crecimiento_promedio:.1f}%"),
        ("Importador principal", mercado_lider),
    ]
    return kpis


def build_plotly_style(fig):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#1f2937"),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def main():
    df = load_data()
    overview = get_dataset_overview(df)
    model, model_metrics, feature_names = train_rf_model(df)

    st.title("📈 Ecuador Export Insights Dashboard")
    st.caption("Dashboard profesional para análisis exploratorio, detección de outliers y predicción de exportaciones usando Random Forest.")

    with st.sidebar:
        st.header("Menú")
        page = st.radio(
            "Seleccione una sección",
            ["Dashboard descriptivo", "Detección de outliers", "Predicción con Random Forest"],
        )

        st.markdown("---")
        st.subheader("Filtros globales")
        importadores = sorted(df["Importador"].unique())
        productos = sorted(df["Producto"].unique())

        selected_importadores = st.multiselect(
            "Importadores",
            importadores,
            default=importadores[:10],
        )
        selected_productos = st.multiselect(
            "Productos",
            productos,
            default=productos,
        )

    filtered = df[
        df["Importador"].isin(selected_importadores)
        & df["Producto"].isin(selected_productos)
    ].copy()

    if filtered.empty:
        st.warning("No hay registros para los filtros seleccionados.")
        return

    if page == "Dashboard descriptivo":
        st.subheader("Resumen ejecutivo")
        st.markdown("Este panel resume la estructura del dataset y los principales indicadores comerciales para elegir piezas clave de análisis.")

        st.markdown("### Análisis exploratorio inicial")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Filas", overview["filas"])
        col2.metric("Columnas", overview["columnas"])
        col3.metric("Importadores únicos", overview["importadores_unicos"])
        col4.metric("Productos únicos", overview["productos_unicos"])

        st.markdown("### KPI principales")
        cards = build_descriptive_cards(filtered)
        c1, c2, c3, c4, c5 = st.columns(5)
        for idx, (label, value) in enumerate(cards):
            cols = [c1, c2, c3, c4, c5]
            cols[idx].metric(label, value)

        st.markdown("### Gráfica 1: Exportación total por importador")
        top_importadores = filtered.groupby("Importador", as_index=False)["Valor exportado ($)"].sum().sort_values("Valor exportado ($)", ascending=False).head(12)
        fig = px.bar(
            top_importadores,
            x="Importador",
            y="Valor exportado ($)",
            color="Valor exportado ($)",
            color_continuous_scale="Sunset",
            title="Top importadores por valor exportado",
        )
        fig = build_plotly_style(fig)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("La gráfica muestra los importadores que concentran mayor valor exportado. Permite detectar destinos prioritarios y rangos de concentración.")

        st.markdown("### Gráfica 2: Participación por producto")
        productos_valor = filtered.groupby("Producto", as_index=False)["Valor exportado ($)"].sum().sort_values("Valor exportado ($)", ascending=False)
        fig2 = px.pie(
            productos_valor,
            names="Producto",
            values="Valor exportado ($)",
            title="Composición de exportaciones por producto",
            hole=0.45,
            color_discrete_sequence=px.colors.sequential.Sunset,
            hover_data={"Producto": True, "Valor exportado ($)": True},
        )
        fig2.update_traces(
            textinfo="percent+label",
            textposition="inside",
            insidetextorientation="radial",
            marker=dict(line=dict(color="#ffffff", width=2)),
            pull=[0.03] + [0] * (len(productos_valor) - 1),
        )
        fig2.update_layout(
            template="plotly_white",
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font=dict(color="#1f2937"),
            margin=dict(l=10, r=10, t=50, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            width=900,
            height=500,
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("El pie chart permite visualizar la composición del total exportado por categoría y confirmar qué productos dominan la estructura del negocio.")

        st.markdown("### Gráfica 3: Relación entre cuota mundial y valor exportado")
        scatter_df = filtered[["Importador", "Producto", "Valor exportado ($)", "Cuota del importador en las importaciones mundiales (%)", "Participación en las exportaciones de Ecuador (%)"]].copy()
        fig3 = px.scatter(
            scatter_df,
            x="Cuota del importador en las importaciones mundiales (%)",
            y="Valor exportado ($)",
            size="Participación en las exportaciones de Ecuador (%)",
            color="Producto",
            hover_name="Importador",
            title="Valor exportado vs cuota mundial",
            color_discrete_sequence=px.colors.sequential.Sunset,
        )
        fig3 = build_plotly_style(fig3)
        st.plotly_chart(fig3, use_container_width=True)
        st.caption("Este scatter muestra cómo la cuota del importador en el mercado mundial se asocia con el valor exportado. Ayuda a identificar oportunidades en países con alta representatividad.")

        st.markdown("### Gráfica 4: Distribución del valor exportado")
        fig4 = px.histogram(
            filtered,
            x="Valor exportado ($)",
            nbins=25,
            color="Producto",
            title="Distribución del valor exportado",
            color_discrete_sequence=px.colors.sequential.Sunset,
        )
        fig4 = build_plotly_style(fig4)
        st.plotly_chart(fig4, use_container_width=True)
        st.caption("El histograma describe la dispersión del valor exportado y ayuda a entender si la distribución está concentrada en pocas transacciones o es más equilibrada.")

        st.markdown("### Tabla de resumen por producto")
        prod_summary = (
            filtered.groupby("Producto")
            .agg(
                Valor_total=("Valor exportado ($)", "sum"),
                Cantidad_total=("Cantidad exportada", "sum"),
                Crecimiento_promedio=("Crecimiento anual de las exportaciones (2024-2025) (%)", "mean"),
                Participación_promedio=("Participación en las exportaciones de Ecuador (%)", "mean"),
            )
            .sort_values("Valor_total", ascending=False)
            .reset_index()
        )
        st.dataframe(prod_summary, use_container_width=True)

    elif page == "Detección de outliers":
        st.subheader("Detección de outliers")
        st.markdown("La detección de outliers se realiza con el método IQR, que identifica registros fuera del rango intercuartílico. Son valores que se alejan mucho de la distribución y pueden indicar anomalías o oportunidades estratégicas.")

        metric = st.selectbox(
            "Seleccione la métrica para analizar outliers",
            ["Valor exportado ($)", "Cantidad exportada", "Valor unitario"],
        )

        q1, q3, iqr, lower, upper, outliers = detect_outliers(filtered, metric)

        col1, col2, col3 = st.columns(3)
        col1.metric("Q1", f"{q1:,.2f}")
        col2.metric("Q3", f"{q3:,.2f}")
        col3.metric("IQR", f"{iqr:,.2f}")

        st.markdown(f"Rango aceptable (IQR): [{lower:,.2f}, {upper:,.2f}]")

        if outliers.empty:
            st.info("No se detectaron outliers para la métrica seleccionada.")
        else:
            st.markdown("### Registros outlier detectados")
            st.dataframe(outliers.sort_values(metric, ascending=False).head(25), use_container_width=True)

        fig_box = px.box(
            filtered,
            x="Producto",
            y=metric,
            color="Producto",
            title=f"Boxplot de {metric} por producto",
            color_discrete_sequence=px.colors.sequential.Sunset,
        )
        fig_box = build_plotly_style(fig_box)
        st.plotly_chart(fig_box, use_container_width=True)
        st.caption("El boxplot muestra la dispersión central, la mediana y los valores extremos. Los puntos aislados representan registros atípicos que merecen revisión operacional o comercial.")

    else:
        st.subheader("Predicción con Random Forest")
        st.markdown("Se seleccionan variables clave del análisis exploratorio para entrenar un modelo de regresión basado en Random Forest. El objetivo es estimar el valor exportado esperado para un importador y producto determinados.")

        product_input = st.selectbox("Producto", sorted(df["Producto"].unique()))
        importer_input = st.selectbox("Importador", sorted(df["Importador"].unique()))

        record = df[(df["Producto"] == product_input) & (df["Importador"] == importer_input)]
        if record.empty:
            st.warning("No existe una combinación exacta producto-importador en la base. A continuación puede ingresar los valores directamente para obtener una predicción.")
            record = pd.DataFrame([df.iloc[0]])

        record = record.iloc[0].copy()

        st.markdown("### Inputs del usuario")
        col1, col2, col3 = st.columns(3)
        with col1:
            cantidad = st.slider("Cantidad exportada", min_value=0, max_value=int(df["Cantidad exportada"].max()), value=int(record["Cantidad exportada"]))
            valor_unitario = st.slider("Valor unitario", min_value=0, max_value=int(df["Valor unitario"].max() + 1), value=int(record["Valor unitario"]))
        with col2:
            participacion_ecuador = st.slider("Participación en Ecuador (%)", min_value=0.0, max_value=100.0, value=float(record["Participación en las exportaciones de Ecuador (%)"]))
            cuota_mundial = st.slider("Cuota mundial (%)", min_value=0.0, max_value=100.0, value=float(record["Cuota del importador en las importaciones mundiales (%)"]))
        with col3:
            crecimiento_24_25 = st.slider("Crecimiento 2024-2025 (%)", min_value=-100.0, max_value=100.0, value=float(record["Crecimiento anual de las exportaciones (2024-2025) (%)"]))
            crecimiento_socios = st.slider("Crecimiento socios 2021-2025 (%)", min_value=-100.0, max_value=100.0, value=float(record["Crecimiento anual del total de importaciones de socios (2021-2025) (%)"]))

        ml_input_df = pd.DataFrame(
            [{
                "Cantidad exportada": cantidad,
                "Valor unitario": valor_unitario,
                "Participación en las exportaciones de Ecuador (%)": participacion_ecuador,
                "Participación de Ecuador en las importaciones de socios (%)": float(record["Participación de Ecuador en las importaciones de socios (%)"]),
                "Cuota del importador en las importaciones mundiales (%)": cuota_mundial,
                "Clasificación del importador en importaciones mundiales": float(record["Clasificación del importador en importaciones mundiales"]),
                "Crecimiento anual de las exportaciones (2024-2025) (%)": crecimiento_24_25,
                "Crecimiento anual del total de importaciones de socios (2021-2025) (%)": crecimiento_socios,
                "Crecimiento anual en cantidad (2021-2025)": float(record["Crecimiento anual en cantidad (2021-2025)"]),
                "Importador": importer_input,
                "Producto": product_input,
            }]
        )

        ml_input_df = pd.get_dummies(ml_input_df, columns=["Importador", "Producto"], drop_first=False)
        missing_cols = [c for c in feature_names if c not in ml_input_df.columns]
        for col in missing_cols:
            ml_input_df[col] = 0

        ml_input_df = ml_input_df.reindex(columns=feature_names, fill_value=0)
        prediction = float(model.predict(ml_input_df)[0])
        actual = float(record["Valor exportado ($)"])

        st.markdown("### Resultado de predicción")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Valor exportado actual", format_currency(actual))
        col_b.metric("Predicción del modelo", format_currency(prediction))
        diff_pct = ((prediction - actual) / actual) * 100 if actual else 0
        col_c.metric("Diferencia estimada", f"{diff_pct:+.1f}%")

        st.info(
            f"Con la combinación seleccionada, el modelo estima un valor exportado de {format_currency(prediction)}. "
            f"Este resultado se basa en la relación entre volumen, cuota de mercado, valoración unitaria y crecimiento reciente."
        )

        st.markdown("### Métricas del modelo")
        st.metric("R²", f"{model_metrics['r2']:.3f}")
        st.metric("MAE", format_currency(model_metrics['mae']))


if __name__ == "__main__":
    main()
