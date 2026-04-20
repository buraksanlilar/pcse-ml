"""
PCSE Yield Prediction — Streamlit Dashboard

Çalıştır:
  streamlit run src/05_streamlit_dashboard.py

Gereksinimler:
  pip install streamlit plotly shap lightgbm joblib pandas numpy matplotlib
"""

import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import shap
import joblib
import streamlit as st

# Proje kök dizinini path'e ekle
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

try:
    from inference_pipeline import YieldPredictor, BOLGE_META, TBASE
except ImportError:
    # inference_pipeline farklı isimle kaydedildiyse
    from predict import YieldPredictor, BOLGE_META, TBASE  # type: ignore

DATA_DIR  = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

# ─── Sayfa ayarları ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PCSE Verim Tahmini",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f0f4f8;
        border-radius: 10px;
        padding: 16px 20px;
        margin: 6px 0;
        border-left: 4px solid #2e86ab;
    }
    .metric-val  { font-size: 2rem; font-weight: 700; color: #2e86ab; }
    .metric-label{ font-size: 0.85rem; color: #555; margin-bottom: 2px; }
    .winner-badge{
        background: #d4edda; border: 1px solid #28a745;
        border-radius: 8px; padding: 8px 16px;
        font-weight: 600; color: #155724; display: inline-block;
    }
    .warn-badge{
        background: #fff3cd; border: 1px solid #ffc107;
        border-radius: 8px; padding: 8px 16px;
        font-weight: 600; color: #856404; display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


# ─── Cache: model & veri yükle ───────────────────────────────────────────────
@st.cache_resource
def load_predictor():
    return YieldPredictor()

@st.cache_data
def load_data():
    df = pd.read_parquet(DATA_DIR / "ml_dataset_multiyear.parquet")
    return df

@st.cache_data
def load_leaderboard():
    p = MODEL_DIR / "leaderboard.csv"
    return pd.read_csv(p) if p.exists() else None

@st.cache_resource
def load_shap_explainer(_model):
    return shap.TreeExplainer(_model)

@st.cache_data(show_spinner=False)
def compute_single_shap_cached(model_path: str, feature_cols: tuple, feature_values: tuple):
    model = joblib.load(model_path)
    explainer = shap.TreeExplainer(model)
    x_row = pd.DataFrame([dict(zip(feature_cols, feature_values))])
    shap_vals = explainer.shap_values(x_row)
    if isinstance(shap_vals, list):
        return shap_vals[0][0].tolist()
    return shap_vals[0].tolist()


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Sunflower_as_food.jpg/240px-Sunflower_as_food.jpg",
             use_column_width=True)
    st.title("🌾 PCSE Verim Tahmini")
    st.markdown("---")

    page = st.radio(
        "Sayfa",
        ["🎯 Tek Tahmin", "📊 Karşılaştırma", "🗺️ Harita", "🏆 Leaderboard", "ℹ️ Hakkında"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.caption("Model: LightGBM / XGBoost / CatBoost")
    st.caption("Veri: PCSE Crop Simulation")


# ══════════════════════════════════════════════════════════════════════════════
# Yükle
# ══════════════════════════════════════════════════════════════════════════════
predictor  = load_predictor()
df_full    = load_data()
leaderboard = load_leaderboard()

districts = sorted(BOLGE_META.keys())
crops     = sorted(TBASE.keys())


# ══════════════════════════════════════════════════════════════════════════════
# Sayfa 1 — Tek Tahmin
# ══════════════════════════════════════════════════════════════════════════════
if page == "🎯 Tek Tahmin":
    st.header("🎯 Verim Tahmini")
    st.markdown("Bölge, ürün ve meteoroloji verilerini girin → anlık tahmin alın.")

    col1, col2, col3 = st.columns([1.2, 1.2, 1])

    with col1:
        st.subheader("📍 Konum & Ürün")
        district = st.selectbox("Bölge", districts)
        crop     = st.selectbox("Ürün", crops)
        variety  = st.text_input("Çeşit (opsiyonel)", value="Default")
        dvs      = st.slider("DVS (Gelişim Aşaması)", -0.1, 2.0, 1.0, 0.05)
        gdd_cum  = st.number_input("Kümülatif GDD", 0, 5000, 1000, 50)
        prec_cum = st.number_input("Kümülatif Yağış (mm)", 0, 2000, 150, 10)
        doy      = st.slider("Yılın Günü", 1, 365, 150)

    with col2:
        st.subheader("🌤️ Meteoroloji")
        t_mean  = st.slider("Ort. Sıcaklık (°C)", -5.0, 45.0, 18.0, 0.5)
        t_min   = st.slider("Min Sıcaklık (°C)", -15.0, 35.0, 8.0, 0.5)
        t_max   = st.slider("Max Sıcaklık (°C)", 0.0, 50.0, 28.0, 0.5)
        humid   = st.slider("Ort. Nem (%)", 10.0, 100.0, 55.0, 1.0)
        precip  = st.slider("Günlük Yağış (mm)", 0.0, 50.0, 2.0, 0.5)
        soil_m  = st.slider("Toprak Nemi (0-1)", 0.0, 0.6, 0.28, 0.01)
        soil_t  = st.slider("Toprak Sıc. (°C)", 0.0, 40.0, 15.0, 0.5)

    with col3:
        st.subheader("🌿 PCSE Durumu (opsiyonel)")
        lai  = st.number_input("LAI",  0.0, 10.0, 3.0, 0.1)
        tagp = st.number_input("TAGP", 0.0, 30000.0, 5000.0, 100.0)
        twlv = st.number_input("TWLV", 0.0, 10000.0, 800.0, 50.0)
        twst = st.number_input("TWST", 0.0, 10000.0, 400.0, 50.0)
        twrt = st.number_input("TWRT", 0.0, 5000.0,  300.0, 50.0)
        sm   = st.number_input("SM",   0.0, 0.6,     0.28,  0.01)

    st.markdown("---")

    if st.button("🚀 Tahmin Et", type="primary", use_container_width=True):
        params = dict(
            variety_name=variety, dvs=dvs, gdd_cumsum=gdd_cum, precip_cumsum=prec_cum,
            day_of_year=int(doy), month=(pd.Timestamp("2024-01-01") + pd.Timedelta(days=int(doy)-1)).month,
            week=int((doy // 7) + 1),
            air_temp_mean=t_mean, air_temp_min=t_min, air_temp_max=t_max,
            air_humidity_mean=humid, air_humidity_min=humid*0.6, air_humidity_max=humid*1.3,
            precip_sum=precip, soil_temp_mean=soil_t, soil_moisture_mean=soil_m,
            lai=lai, tagp=tagp, twlv=twlv, twst=twst, twrt=twrt, sm=sm,
        )

        with st.spinner("Hesaplanıyor..."):
            result = predictor.predict_single(district, crop, **params)
            unc    = predictor.predict_with_uncertainty(district, crop, **params)

        # Metrik kartları
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Tahmin Verim</div>
                <div class="metric-val">{result['twso_pred']:,.0f}</div>
                <div class="metric-label">kg/ha</div></div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">%80 Alt Sınır</div>
                <div class="metric-val">{unc['ci_lower']:,.0f}</div>
                <div class="metric-label">kg/ha</div></div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">%80 Üst Sınır</div>
                <div class="metric-val">{unc['ci_upper']:,.0f}</div>
                <div class="metric-label">kg/ha</div></div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Büyüme Dönemi</div>
                <div class="metric-val" style="font-size:1.3rem">{result['growth_stage'].title()}</div>
                <div class="metric-label">DVS={dvs:.2f}</div></div>""", unsafe_allow_html=True)

        # Güven aralığı çubuğu
        st.markdown("#### Tahmin Güven Aralığı")
        fig_ci = go.Figure(go.Bar(
            x=[result['twso_pred']], y=["Tahmin"],
            orientation='h', marker_color='#2e86ab',
            error_x=dict(
                type='data', symmetric=False,
                array=[unc['ci_upper'] - result['twso_pred']],
                arrayminus=[result['twso_pred'] - unc['ci_lower']],
                color='#555'
            )
        ))
        fig_ci.update_layout(height=150, margin=dict(l=0, r=0, t=10, b=10),
                              xaxis_title="twso_final (kg/ha)")
        st.plotly_chart(fig_ci, use_container_width=True)

        # SHAP tek örnek
        st.markdown("#### Feature Katkıları (SHAP)")
        try:
            feat_row    = predictor._build_row({"district_name": district, "crop_name": crop, **params})
            feat_vals = tuple(float(v) for v in feat_row.iloc[0].values)
            shap_vec = np.array(
                compute_single_shap_cached(
                    str(predictor.model_path),
                    tuple(predictor.feature_cols),
                    feat_vals,
                )
            )
            top_idx     = np.argsort(np.abs(shap_vec))[-15:][::-1]
            top_feats   = [predictor.feature_cols[i] for i in top_idx]
            top_vals    = shap_vec[top_idx]
            colors      = ['#2e86ab' if v > 0 else '#e84855' for v in top_vals]

            fig_shap, ax = plt.subplots(figsize=(8, 4))
            ax.barh(top_feats[::-1], top_vals[::-1], color=colors[::-1])
            ax.axvline(0, color='black', lw=0.8)
            ax.set_title("SHAP — Top 15 Feature Katkısı")
            ax.set_xlabel("SHAP Değeri")
            plt.tight_layout()
            st.pyplot(fig_shap)
        except Exception as e:
            st.info(f"SHAP hesaplanamadı: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# Sayfa 2 — Crop Karşılaştırma
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Karşılaştırma":
    st.header("📊 Ürün & Bölge Karşılaştırması")

    tab1, tab2, tab3 = st.tabs(["Ürün Bazlı", "Bölge Bazlı", "Zaman Serisi"])

    with tab1:
        combo_stats = (
            df_full.groupby(["crop_name", "district_name", "year"])["twso_final"]
            .first()
            .reset_index()
            .groupby("crop_name")["twso_final"]
            .agg(["mean","std","min","max","count"])
            .round(1)
            .sort_values("mean", ascending=False)
            .reset_index()
        )
        fig_box = px.box(
            df_full.groupby(["crop_name","district_name","variety_name","year"])["twso_final"].first().reset_index(),
            x="crop_name", y="twso_final", color="crop_name",
            title="Ürün Bazlı Verim Dağılımı",
            labels={"twso_final": "Verim (kg/ha)", "crop_name": "Ürün"},
        )
        fig_box.update_layout(showlegend=False, xaxis_tickangle=-40)
        st.plotly_chart(fig_box, use_container_width=True)
        st.dataframe(combo_stats, use_container_width=True)

    with tab2:
        dist_stats = (
            df_full.groupby(["district_name","crop_name","year"])["twso_final"]
            .first().reset_index()
            .groupby("district_name")["twso_final"]
            .agg(["mean","count"])
            .round(1)
            .sort_values("mean", ascending=False)
            .reset_index()
        )
        fig_dist = px.bar(
            dist_stats, x="district_name", y="mean",
            title="Bölge Bazlı Ortalama Verim",
            labels={"mean": "Ort. Verim (kg/ha)", "district_name": "Bölge"},
            color="mean", color_continuous_scale="Blues"
        )
        fig_dist.update_layout(xaxis_tickangle=-40)
        st.plotly_chart(fig_dist, use_container_width=True)

    with tab3:
        sel_crop = st.selectbox("Ürün seç", crops, key="ts_crop")
        sel_dist = st.selectbox("Bölge seç", districts, key="ts_dist")
        ts_df = df_full[
            (df_full["crop_name"] == sel_crop) &
            (df_full["district_name"] == sel_dist)
        ].sort_values("date")

        if len(ts_df) > 0:
            cols_to_plot = st.multiselect(
                "Gösterilecek değişkenler",
                ["DVS","LAI","TAGP","AIR_TEMP_mean","PRECIP_sum","SM"],
                default=["DVS","LAI"]
            )
            for col in cols_to_plot:
                if col in ts_df.columns:
                    fig_ts = px.line(ts_df, x="date", y=col,
                                     title=f"{col} — {sel_crop} / {sel_dist}")

                    if "DVS" in ts_df.columns:
                        for dvs_point in [0.0, 1.0]:
                            hit = ts_df[ts_df["DVS"] >= dvs_point]
                            if len(hit) > 0:
                                cross_date = hit.iloc[0]["date"]
                                fig_ts.add_vline(
                                    x=cross_date,
                                    line_dash="dash",
                                    line_color="#444",
                                    annotation_text=f"DVS={dvs_point:.0f}",
                                    annotation_position="top right",
                                )
                    st.plotly_chart(fig_ts, use_container_width=True)
        else:
            st.warning("Bu kombinasyon için veri bulunamadı.")


# ══════════════════════════════════════════════════════════════════════════════
# Sayfa 3 — Harita
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🗺️ Harita":
    st.header("🗺️ Türkiye Verim Haritası")

    sel_crop_map = st.selectbox("Ürün seç", crops)

    map_data = (
        df_full[df_full["crop_name"] == sel_crop_map]
        .groupby(["district_name","year"])["twso_final"]
        .first()
        .reset_index()
        .groupby("district_name")["twso_final"]
        .mean()
        .round(1)
        .reset_index()
    )
    map_data = map_data.merge(
        pd.DataFrame([{"district_name": k, "lat": v["latitude"], "lon": v["longitude"]}
                      for k, v in BOLGE_META.items()]),
        on="district_name"
    )

    fig_map = px.scatter_mapbox(
        map_data,
        lat="lat", lon="lon",
        size="twso_final", color="twso_final",
        hover_name="district_name",
        hover_data={"twso_final": ":.1f", "lat": False, "lon": False},
        color_continuous_scale="YlOrRd",
        size_max=35,
        zoom=5.5,
        center={"lat": 39.0, "lon": 35.0},
        mapbox_style="carto-positron",
        title=f"{sel_crop_map.title()} — Bölgesel Verim (twso_final)"
    )
    fig_map.update_layout(height=550, margin={"r":0,"t":40,"l":0,"b":0})
    st.plotly_chart(fig_map, use_container_width=True)

    st.dataframe(
        map_data[["district_name","twso_final"]].sort_values("twso_final", ascending=False)
        .rename(columns={"district_name":"Bölge","twso_final":"Verim (kg/ha)"})
        .reset_index(drop=True),
        use_container_width=True
    )


# ══════════════════════════════════════════════════════════════════════════════
# Sayfa 4 — Leaderboard
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🏆 Leaderboard":
    st.header("🏆 Model Leaderboard")

    if leaderboard is not None:
        winner = leaderboard.iloc[0]["Model"] if "Model" in leaderboard.columns else "—"
        st.markdown(f'<div class="winner-badge">🥇 Kazanan Model: {winner}</div>', unsafe_allow_html=True)
        st.markdown("")

        st.dataframe(leaderboard.style.highlight_max(subset=["R2"], color="#d4edda")
                                      .highlight_min(subset=["RMSE","MAE"], color="#d4edda"),
                     use_container_width=True)

        fig_lb = px.bar(
            leaderboard, x="Model", y="R2", color="R2",
            color_continuous_scale="Blues",
            title="Model R² Karşılaştırması",
            text_auto=".3f"
        )
        fig_lb.update_layout(yaxis_range=[0, 1.05])
        st.plotly_chart(fig_lb, use_container_width=True)
    else:
        st.info("Henüz leaderboard yok. 03_model_comparison.ipynb'i çalıştırın.")


# ══════════════════════════════════════════════════════════════════════════════
# Sayfa 5 — Hakkında
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ Hakkında":
    st.header("ℹ️ Proje Hakkında")
    st.markdown("""
    ### PCSE Verim Tahmin Sistemi

    Bu proje, **PCSE (Python Crop Simulation Environment)** simülasyon çıktılarını kullanarak
    makine öğrenmesi tabanlı ürün verimi tahmini yapmaktadır.

    #### Veri Kaynağı
    - 15 farklı Türkiye bölgesi
    - 19 farklı ürün (geçerli)
    - Saatlik meteoroloji + PCSE durum değişkenleri
    - Tarih aralığı: 2014-01-01 → 2024-12-31 (11 yıl)

    #### Pipeline
    | Adım | Dosya | İçerik |
    |------|-------|--------|
    | 01 | `pipeline_01.py` | Temizleme + Feature Engineering |
    | 02 | `02_model_lightgbm.ipynb` | LightGBM + Optuna Tuning |
    | 03 | `03_model_comparison.ipynb` | Model Karşılaştırması |
    | 04 | `inference_pipeline.py` | Tahmin Servisi |
    | 05 | `05_streamlit_dashboard.py` | Bu Dashboard |

    #### Hedef Değişken
    `twso_final` — Final tane ağırlığı (kg/ha eşdeğeri)

    #### Teknolojiler
    `LightGBM` · `XGBoost` · `CatBoost` · `Optuna` · `SHAP` · `Streamlit` · `Plotly`
    """)