"""
MDF Üretim Süresi Tahmin Sistemi
Streamlit Web Uygulaması — Bitirme Projesi / Kongre Sunumu
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle, os, sys
import warnings
warnings.filterwarnings("ignore")

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ─── SAYFA AYARLARI ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MDF Üretim Süresi Tahmin Sistemi",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Sidebar Arka Planı */
/* 1. Arka planı daha açık ve canlı yapıyoruz */
    .stApp {
        background: linear-gradient(rgba(255, 255, 255, 0.7), rgba(255, 255, 255, 0.7)), 
                    url("https://www.masplus.com.tr/wp-content/uploads/2019/10/uretim-yonetim-fabrika.webp");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.metric-card {
    background: linear-gradient(135deg, #1e3a5f 0%, #16213e 100%);
    border-radius: 12px; padding: 20px; text-align: center;
    border: 1px solid #2a5298; margin-bottom: 10px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}
.metric-card .value { font-size: 2rem; font-weight: 700; color: #4fc3f7; }
.metric-card .label { font-size: 0.85rem; color: #90caf9; margin-top: 4px; }
.metric-card .sub   { font-size: 0.75rem; color: #64b5f6; margin-top: 2px; }

.header-box {
    background: linear-gradient(90deg, #0d47a1, #1565c0, #1976d2);
    border-radius: 12px; padding: 24px 32px; margin-bottom: 24px;
    box-shadow: 0 4px 20px rgba(13,71,161,0.4);
}
.header-box h1 { color: white; font-size: 1.9rem; font-weight: 700; margin: 0; }
.header-box p  { color: #bbdefb; font-size: 0.95rem; margin: 6px 0 0 0; }

.prediction-box {
    background: linear-gradient(135deg, #1b5e20, #2e7d32);
    border-radius: 12px; padding: 28px; text-align: center;
    border: 2px solid #43a047;
    box-shadow: 0 6px 20px rgba(27,94,32,0.4);
}
.prediction-box .val { font-size: 3.5rem; font-weight: 700; color: #a5d6a7; }
.prediction-box .lbl { font-size: 1rem; color: #c8e6c9; margin-top: 8px; }

.info-tag {
    display:inline-block; background:#1565c0; color:white;
    border-radius:6px; padding:3px 10px; font-size:0.78rem; margin:2px;
}
.section-title {
    font-size: 1.15rem; font-weight: 600; color: #1976d2;
    border-left: 4px solid #1976d2; padding-left: 12px; margin: 20px 0 12px 0;
}
</style>
""", unsafe_allow_html=True)

# ─── YARDIMCI FONKSİYONLAR ──────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_artifacts():
    # Mevcut dosyanın (süre2.py) klasörünü otomatik bulur
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # "model_artifacts" klasörüne giden yolu güvenli şekilde oluşturur
    artifacts_path = os.path.join(base_dir, "model_artifacts")
    
    # Dosyaları bu yol üzerinden aç
    with open(os.path.join(artifacts_path, "models.pkl"), "rb") as f: 
        models = pickle.load(f)
    with open(os.path.join(artifacts_path, "encoders.pkl"), "rb") as f: 
        encoders = pickle.load(f)
    with open(os.path.join(artifacts_path, "feature_cols.pkl"), "rb") as f: 
        feat_cols = pickle.load(f)
    with open(os.path.join(artifacts_path, "shap_data.pkl"), "rb") as f: 
        shap_data = pickle.load(f)
    with open(os.path.join(artifacts_path, "results_df.pkl"), "rb") as f: 
        results = pickle.load(f)
        
    return models, encoders, feat_cols, shap_data, results

@st.cache_data(show_spinner=False)
def load_raw_data():
    # Klasör yolunu al
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Excel dosyası süre2.py ile aynı klasörde olmalı
    path = os.path.join(base_dir, "MDF_Uretim_Veri_Sistemi.xlsx")
    
    return pd.read_excel(path, sheet_name="Uretim_Kayitlari")

def preprocess_input(row_dict, encoders, feat_cols):
    cat_map = {
        "Vardiya": "Vardiya_enc",
        "Urun_Tipi": "Urun_Tipi_enc",
        "Hammadde_Tipi": "Hammadde_Tipi_enc",
        "Ariza_Tipi": "Ariza_Tipi_enc",
        "Zimpara_Kum": "Zimpara_Kum_enc",
    }
    for cat, enc_col in cat_map.items():
        le = encoders[cat]
        val = row_dict.get(cat, le.classes_[0])
        if val in le.classes_:
            row_dict[enc_col] = int(le.transform([val])[0])
        else:
            row_dict[enc_col] = 0

    # Tarih öznitelikleri (varsayılan)
    row_dict.setdefault("Ay", 6)
    row_dict.setdefault("HaftaGunu", 0)
    row_dict.setdefault("Hafta", 26)

    X = pd.DataFrame([{c: row_dict.get(c, 0) for c in feat_cols}])
    return X

# ─── RENK PALETİ ─────────────────────────────────────────────────────────────
COLORS = {
    "XGBoost":     "#f97316",
    "LightGBM":    "#8b5cf6",
    "RandomForest":"#22c55e",
}

# ══════════════════════════════════════════════════════════════════════════════
# ANA UYGULAMA
# ══════════════════════════════════════════════════════════════════════════════

def main():
    models, encoders, feat_cols, shap_data, results_df = load_artifacts()
    df = load_raw_data()

    # Tarih işleme
    df["Tarih"] = pd.to_datetime(df["Tarih"])

    # ─── HEADER ─────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="header-box">
        <h1>🏭 MDF Üretim Süresi Tahmin Sistemi</h1>
        <p>Makine Öğrenmesi Tabanlı Üretim Çevrim Süresi Optimizasyonu · XGBoost · LightGBM · Random Forest</p>
    </div>
    """, unsafe_allow_html=True)

    # ─── SIDEBAR ────────────────────────────────────────────────────────────
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/38/Info_Simple.svg/48px-Info_Simple.svg.png", width=36)
        st.markdown("### Navigasyon")
        page = st.radio("", [
            "📊 Veri Analizi",
            "🤖 Model Performansı",
            "🎯 Canlı Tahmin",
            "🔍 SHAP Açıklanabilirlik",
        ], label_visibility="collapsed")

        st.markdown("---")
        st.markdown("**Veri Seti Özeti**")
        st.markdown(f"""
        <div style='font-size:0.82rem; color:#ccc;'>
        • <b>{len(df):,}</b> vardiya kaydı<br>
        • <b>{df['Tarih'].min().strftime('%d.%m.%Y')}</b> – <b>{df['Tarih'].max().strftime('%d.%m.%Y')}</b><br>
        • <b>{df['Urun_Tipi'].nunique()}</b> ürün tipi<br>
        • <b>54</b> öznitelik → <b>{len(feat_cols)}</b> model girdisi<br>
        • Hedef: <i>Net Çalışma Süresi</i> (dk)
        </div>
        """, unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════
    # SAYFA 1: VERİ ANALİZİ
    # ════════════════════════════════════════════════════════════════════════
    if page == "📊 Veri Analizi":
        st.markdown('<div class="section-title">Veri Seti Genel Bakış</div>', unsafe_allow_html=True)

        # KPI kartları
        c1, c2, c3, c4, c5 = st.columns(5)
        kpis = [
            (c1, f"{len(df):,}", "Toplam Kayıt", "Vardiya"),
            (c2, f"{df['Net_Calisma_Sure_dk'].mean():.1f} dk", "Ort. Net Çalışma", "Süresi / Vardiya"),
            (c3, f"{df['Fire_Oran_Pct'].mean():.1f}%", "Ortalama Fire", "Oranı"),
            (c4, f"{df['Kalite_Skoru'].mean():.1f}", "Ort. Kalite", "Skoru (0–100)"),
            (c5, f"{df['Pres_Uptime_Pct'].mean():.1f}%", "Pres Uptime", "Yüzdesi"),
        ]
        for col, val, lbl, sub in kpis:
            col.markdown(f"""
            <div class="metric-card">
                <div class="value">{val}</div>
                <div class="label">{lbl}</div>
                <div class="sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-title">Üretim Süresi Dağılımı & Zaman Trendi</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            fig = px.histogram(df, x="Net_Calisma_Sure_dk", nbins=50, color="Vardiya",
                               title="Net Çalışma Süresi Dağılımı (Vardiyaya Göre)",
                               labels={"Net_Calisma_Sure_dk": "Süre (dk)", "count": "Frekans"},
                               color_discrete_sequence=["#3b82f6","#f97316","#22c55e"],
                               template="plotly_dark")
            fig.update_layout(legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            urun_stats = df.groupby("Urun_Tipi")["Net_Calisma_Sure_dk"].agg(["mean","std"]).reset_index()
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                x=urun_stats["Urun_Tipi"], y=urun_stats["mean"],
                error_y=dict(type='data', array=urun_stats["std"], visible=True),
                marker_color=["#3b82f6","#f97316","#22c55e","#a855f7","#ef4444"],
                name="Ort. Süre"
            ))
            fig2.update_layout(title="Ürün Tipine Göre Ort. Çalışma Süresi (±1σ)",
                               xaxis_title="Ürün Tipi", yaxis_title="Süre (dk)",
                               template="plotly_dark")
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown('<div class="section-title">Arıza Analizi & Korelasyon</div>', unsafe_allow_html=True)
        col3, col4 = st.columns(2)

        with col3:
            ariza = df[df["Ariza_Tipi"] != "Yok"].groupby("Ariza_Tipi")["Ariza_Sure_dk"].agg(["sum","count"]).reset_index()
            ariza.columns = ["Ariza_Tipi","Toplam_Sure","Adet"]
            ariza = ariza.sort_values("Toplam_Sure", ascending=True)
            fig3 = px.bar(ariza, x="Toplam_Sure", y="Ariza_Tipi", orientation="h",
                          color="Adet", color_continuous_scale="Reds",
                          title="Arıza Türlerine Göre Toplam Kayıp Süre",
                          labels={"Toplam_Sure":"Toplam Kayıp (dk)"},
                          template="plotly_dark")
            st.plotly_chart(fig3, use_container_width=True)

        with col4:
            key_vars = ["Net_Calisma_Sure_dk", "Pres_Sure_sn_adet", "Ariza_Sure_dk",
                        "Kurutma_Sure_dk", "Pres_Uptime_Pct", "Lif_Kalite_Skoru",
                        "Nem_Orani_Giris_Pct", "Recine_Tuketim_kg_m3"]
            corr = df[key_vars].corr()
            labels_short = ["NetSüre","PresSüre","ArızaSüre","KurutmaSüre",
                            "PresUptime","LifKalite","NemGiriş","ReçineTük"]
            fig4 = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                             zmin=-1, zmax=1, title="Korelasyon Matrisi (Anahtar Değişkenler)",
                             template="plotly_dark",
                             x=labels_short, y=labels_short)
            fig4.update_layout(height=380)
            st.plotly_chart(fig4, use_container_width=True)

        # Aylık trend
        st.markdown('<div class="section-title">Aylık Üretim Trendi</div>', unsafe_allow_html=True)
        df["AyYil"] = df["Tarih"].dt.to_period("M").astype(str)
        monthly = df.groupby("AyYil").agg(
            OrtSure=("Net_Calisma_Sure_dk","mean"),
            ToplamUretim=("Net_Uretim_Adet","sum"),
            OrtFire=("Fire_Oran_Pct","mean")
        ).reset_index()

        fig5 = make_subplots(specs=[[{"secondary_y": True}]])
        fig5.add_trace(go.Scatter(x=monthly["AyYil"], y=monthly["OrtSure"],
                                  mode="lines+markers", name="Ort. Net Süre (dk)",
                                  line=dict(color="#3b82f6", width=2.5)), secondary_y=False)
        fig5.add_trace(go.Bar(x=monthly["AyYil"], y=monthly["OrtFire"],
                              name="Ort. Fire (%)", marker_color="#ef4444", opacity=0.5),
                       secondary_y=True)
        fig5.update_layout(title="Aylık Ortalama Net Çalışma Süresi & Fire Oranı",
                           template="plotly_dark", height=350,
                           legend=dict(orientation="h", y=-0.2))
        fig5.update_yaxes(title_text="Süre (dk)", secondary_y=False)
        fig5.update_yaxes(title_text="Fire (%)", secondary_y=True)
        st.plotly_chart(fig5, use_container_width=True)

    # ════════════════════════════════════════════════════════════════════════
    # SAYFA 2: MODEL PERFORMANSI
    # ════════════════════════════════════════════════════════════════════════
    elif page == "🤖 Model Performansı":
        st.markdown('<div class="section-title">Model Karşılaştırma Tablosu</div>', unsafe_allow_html=True)

        display_df = results_df[["Model","MAE (dk)","RMSE (dk)","R²","MAPE (%)","CV R² (mean)","CV R² (std)"]].copy()

        # Renk stillemesi
        def style_df(row):
            c = COLORS.get(row["Model"], "#fff")
            return [f"color:{c}; font-weight:600" if i == 0 else "" for i in range(len(row))]

        st.dataframe(display_df.style.apply(style_df, axis=1).format({
            "MAE (dk)": "{:.3f}", "RMSE (dk)": "{:.3f}",
            "R²": "{:.4f}", "MAPE (%)": "{:.3f}%",
            "CV R² (mean)": "{:.4f}", "CV R² (std)": "{:.4f}"
        }), use_container_width=True, hide_index=True)

        # Metrik grafikleri
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(results_df, x="Model", y="R²",
                         color="Model", color_discrete_map=COLORS,
                         title="R² Skoru (↑ yüksek = iyi)",
                         template="plotly_dark",
                         text_auto=".4f")
            fig.update_traces(textposition="outside")
            fig.update_layout(yaxis_range=[0.5, 0.7], showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig2 = px.bar(results_df, x="Model", y="MAE (dk)",
                          color="Model", color_discrete_map=COLORS,
                          title="MAE – Ortalama Mutlak Hata (dk) (↓ düşük = iyi)",
                          template="plotly_dark",
                          text_auto=".3f")
            fig2.update_traces(textposition="outside")
            fig2.update_layout(showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

        # Gerçek vs Tahmin scatter
        st.markdown('<div class="section-title">Gerçek vs Tahmin (Test Seti)</div>', unsafe_allow_html=True)
        pred_path = os.path.dirname(__file__) + "/model_artifacts/test_predictions.csv"
        pred_df = pd.read_csv(pred_path)

        sel_model = st.selectbox("Model seçin:", ["XGBoost", "LightGBM", "RandomForest"])
        col_pred = f"Tahmin_{sel_model}"

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=pred_df["Gercek_Sure"], y=pred_df[col_pred],
            mode="markers",
            marker=dict(color=COLORS[sel_model], size=5, opacity=0.6),
            name=sel_model
        ))
        mn, mx = pred_df["Gercek_Sure"].min(), pred_df["Gercek_Sure"].max()
        fig3.add_trace(go.Scatter(x=[mn,mx], y=[mn,mx], mode="lines",
                                  line=dict(color="white", dash="dash"), name="Mükemmel Tahmin"))
        r2 = r2_score(pred_df["Gercek_Sure"], pred_df[col_pred])
        mae = mean_absolute_error(pred_df["Gercek_Sure"], pred_df[col_pred])
        fig3.update_layout(
            title=f"{sel_model} — Gerçek vs Tahmin | R²={r2:.4f} | MAE={mae:.2f} dk",
            xaxis_title="Gerçek Süre (dk)", yaxis_title="Tahmin (dk)",
            template="plotly_dark", height=420
        )
        st.plotly_chart(fig3, use_container_width=True)

        # Hata dağılımı
        resid = pred_df["Gercek_Sure"] - pred_df[col_pred]
        fig4 = px.histogram(x=resid, nbins=40, title=f"{sel_model} — Artık Hata Dağılımı",
                            labels={"x": "Artık (dk)"}, template="plotly_dark",
                            color_discrete_sequence=[COLORS[sel_model]])
        fig4.add_vline(x=0, line_dash="dash", line_color="white")
        st.plotly_chart(fig4, use_container_width=True)

    # ════════════════════════════════════════════════════════════════════════
    # SAYFA 3: CANLI TAHMİN
    # ════════════════════════════════════════════════════════════════════════
    elif page == "🎯 Canlı Tahmin":
        st.markdown('<div class="section-title">Yeni Sipariş / İş Emri için Üretim Süresi Tahmini</div>', unsafe_allow_html=True)
        st.info("Sol panelden veya aşağıdan parametreleri girin, anında tahmin alın.")

        col_form, col_res = st.columns([1.3, 1])

        with col_form:
            with st.expander("📦 Sipariş Bilgileri", expanded=True):
                c1, c2 = st.columns(2)
                urun_tipi    = c1.selectbox("Ürün Tipi", ["MDF-8mm","MDF-12mm","MDF-16mm","MDF-18mm","MDF-25mm"])
                vardiya      = c2.selectbox("Vardiya", ["Sabah (06:00-14:00)","Öğle (14:00-22:00)","Gece (22:00-06:00)"])
                hammadde     = c1.selectbox("Hammadde Tipi", ["Çam Yongası","Kavak Yongası","Ladin Yongası","Karışık Atık"])
                ariza_tipi   = c2.selectbox("Arıza Tipi", ["Yok","Pres Arızası","Kurutucu Tıkanması","Konveyör Durması","Tutkal Sistemi","Soğutucu Kapasitesi","Elektrik Kesintisi"])
                ariza_sure   = st.slider("Arıza Süresi (dk)", 0, 180, 0, step=5)

            with st.expander("⚙️ Proses Parametreleri", expanded=True):
                c1, c2, c3 = st.columns(3)
                yonga_ton        = c1.number_input("Yonga Giriş (ton)", 450.0, 650.0, 552.0, step=5.0)
                nem_giris        = c2.number_input("Nem Oranı Giriş (%)", 40.0, 70.0, 52.0, step=0.5)
                pres_sicaklik    = c3.number_input("Pres Sıcaklığı (°C)", 150.0, 230.0, 190.0, step=1.0)
                pres_basinc      = c1.number_input("Pres Basıncı (bar)", 20.0, 60.0, 38.0, step=0.5)
                pres_sure        = c2.number_input("Pres Süresi (sn/adet)", 15.0, 60.0, 30.0, step=0.5)
                pres_uptime      = c3.number_input("Pres Uptime (%)", 70.0, 100.0, 92.0, step=0.5)
                kurutma_sick     = c1.number_input("Kurutma Sıcaklığı (°C)", 100.0, 200.0, 150.0, step=1.0)
                kurutma_sure     = c2.number_input("Kurutma Süresi (dk)", 40.0, 100.0, 60.0, step=1.0)
                recine           = c3.number_input("Reçine Tük. (kg/m³)", 50.0, 120.0, 78.0, step=1.0)
                defib_sick       = c1.number_input("Defibratör Sıcaklık (°C)", 160.0, 200.0, 178.0, step=0.5)
                defib_basinc     = c2.number_input("Defibratör Basınç (bar)", 6.0, 12.0, 8.5, step=0.1)
                serme_hizi       = c3.number_input("Serme Hızı (m/dk)", 20.0, 60.0, 38.0, step=0.5)
                sogutma_sure     = c1.number_input("Soğutma Süresi (dk)", 5.0, 40.0, 15.0, step=1.0)
                elektrik         = c2.number_input("Elektrik Tük. (kWh)", 1000.0, 5000.0, 2800.0, step=50.0)
                dogalgaz         = c3.number_input("Doğalgaz Tük. (m³)", 200.0, 1000.0, 520.0, step=10.0)

            model_sec = st.selectbox("Tahmin Modeli", ["XGBoost","LightGBM","RandomForest"])

        # Tahmin
        row = {
            "Vardiya": vardiya, "Urun_Tipi": urun_tipi, "Hammadde_Tipi": hammadde,
            "Ariza_Tipi": ariza_tipi, "Zimpara_Kum": "80",
            "Yonga_Giris_Ton": yonga_ton, "Nem_Orani_Giris_Pct": nem_giris,
            "Yonga_Hazirlama_Sure_dk": 75.0, "Eleme_Sure_dk": 30.0,
            "Ince_Yonga_Pct": 15.0, "Kaba_Yonga_Pct": 20.0, "Red_Yonga_Pct": 5.0,
            "Islatma_Sicaklik_C": 80.0, "Nem_Islatma_Sonrasi_Pct": 65.0,
            "Defibrator_Buhar_Sicaklik_C": defib_sick, "Defibrator_Basinc_Bar": defib_basinc,
            "Lif_Kalite_Skoru": 75.0,
            "Recine_Tuketim_kg_m3": recine, "Sertlestirici_Doz_kg_m3": 5.0, "Tutkal_pH": 7.0,
            "Kurutma_Sicaklik_C": kurutma_sick, "Kurutma_Sure_dk": kurutma_sure,
            "Nem_Kurutma_Sonrasi_Pct": 8.0,
            "Hali_Kalinlik_cm": 20.0, "Serme_Hizi_m_dk": serme_hizi,
            "OnPres_Basinc_Bar": 3.0, "OnPres_Sure_sn": 30.0,
            "Pres_Sure_sn_adet": pres_sure, "Pres_Sicaklik_C": pres_sicaklik,
            "Pres_Basinc_Bar": pres_basinc, "Pres_Uptime_Pct": pres_uptime,
            "Ariza_Sure_dk": ariza_sure,
            "Sogutma_Sure_dk": sogutma_sure, "Sogutma_Cikis_Sicaklik_C": 35.0,
            "Sogutucu_Kapasite_Kullanim_Pct": 75.0,
            "Kalinlik_Sapma_mm": 0.1, "Yuzey_Puruzsuzluk_Ra_um": 1.5,
            "Ebat_Hatasi_mm": 0.3,
            "Elektrik_Tuketim_kWh": elektrik, "Dogalgaz_Tuketim_m3": dogalgaz,
            "Olculen_Kalinlik_mm": 16.0, "Kalinlik_Hedef_mm": 16.0,
        }

        X_input = preprocess_input(row, encoders, feat_cols)
        pred = models[model_sec].predict(X_input)[0]

        with col_res:
            # Tüm model tahminleri
            tum_tahminler = {m: models[m].predict(X_input)[0] for m in models}

            st.markdown(f"""
            <div class="prediction-box">
                <div class="val">{pred:.1f} dk</div>
                <div class="lbl">Tahmini Net Üretim Süresi<br><small>({model_sec})</small></div>
            </div>""", unsafe_allow_html=True)

            st.markdown("**Tüm Model Tahminleri:**")
            for m, v in tum_tahminler.items():
                diff = v - pred
                st.markdown(f"""
                <span class="info-tag" style="background:{COLORS[m]}">{m}: {v:.1f} dk</span>
                """, unsafe_allow_html=True)

            st.markdown("---")
            # Üretim verimliliği tahmini
            toplam_dk = 480  # 8 saatlik vardiya
            verimlilik = (pred / toplam_dk) * 100
            st.markdown(f"""
            **📈 Tahmin Özeti**
            | Metrik | Değer |
            |--------|-------|
            | Net Üretim Süresi | **{pred:.1f} dk** |
            | Toplam Vardiya | 480 dk |
            | Verimlilik | **{verimlilik:.1f}%** |
            | Durma Payı | **{480 - pred:.1f} dk** |
            | Güven Aralığı (±MAE) | **±6.3 dk** |
            """)

            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=pred,
                delta={"reference": df["Net_Calisma_Sure_dk"].mean(), "valueformat": ".1f"},
                gauge={
                    "axis": {"range": [360, 480]},
                    "bar": {"color": COLORS[model_sec]},
                    "steps": [
                        {"range": [360, 420], "color": "#ef4444"},
                        {"range": [420, 455], "color": "#f97316"},
                        {"range": [455, 480], "color": "#22c55e"},
                    ],
                    "threshold": {
                        "line": {"color": "white", "width": 3},
                        "thickness": 0.75,
                        "value": df["Net_Calisma_Sure_dk"].mean()
                    }
                },
                title={"text": "Üretim Süresi (dk)<br><span style='font-size:0.8em'>Beyaz çizgi = tarihsel ort.</span>"},
                number={"suffix": " dk"}
            ))
            fig_gauge.update_layout(template="plotly_dark", height=280, margin=dict(t=60,b=0))
            st.plotly_chart(fig_gauge, use_container_width=True)

    # ════════════════════════════════════════════════════════════════════════
    # SAYFA 4: SHAP AÇIKLANABILIRLIK
    # ════════════════════════════════════════════════════════════════════════
    elif page == "🔍 SHAP Açıklanabilirlik":
        st.markdown('<div class="section-title">SHAP — Model Kararlarının Açıklanabilirliği (XGBoost)</div>', unsafe_allow_html=True)
        st.markdown("""
        SHAP (SHapley Additive exPlanations), her özniteliğin tahmin üzerindeki etkisini oyun teorisi 
        ile hesaplar. Pozitif değer süreyi **uzatır**, negatif değer **kısaltır**.
        """)

        shap_vals = shap_data["shap_values"]
        X_test    = shap_data["X_test"]

        tab1, tab2, tab3 = st.tabs(["📊 Feature Importance", "🌡️ SHAP Beeswarm", "🔬 Tek Tahmin Analizi"])

        with tab1:
            mean_abs = np.abs(shap_vals).mean(axis=0)
            feat_imp = pd.DataFrame({
                "Öznitelik": feat_cols,
                "Ort. |SHAP|": mean_abs
            }).sort_values("Ort. |SHAP|", ascending=True).tail(20)

            fig = px.bar(feat_imp, x="Ort. |SHAP|", y="Öznitelik", orientation="h",
                         title="Top-20 Öznitelik — Ortalama |SHAP| Değeri",
                         color="Ort. |SHAP|", color_continuous_scale="Blues",
                         template="plotly_dark")
            fig.update_layout(height=550, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.info("SHAP beeswarm grafiği oluşturuluyor (matplotlib)…")
            fig_bw, ax = plt.subplots(figsize=(10, 7))
            fig_bw.patch.set_facecolor("#0e1117")
            ax.set_facecolor("#0e1117")

            top_idx = np.argsort(np.abs(shap_vals).mean(0))[-15:]
            shap.summary_plot(
                shap_vals[:, top_idx],
                X_test.iloc[:, top_idx],
                feature_names=[feat_cols[i] for i in top_idx],
                show=False, plot_size=None
            )
            plt.tight_layout()
            st.pyplot(fig_bw, use_container_width=True)
            plt.close()

        with tab3:
            idx = st.slider("Test örneği indeksi:", 0, len(X_test)-1, 0)
            st.markdown(f"**Seçili örnek gerçek süresi:** {shap_data['X_test'].iloc[idx].name}")

            sv = shap_vals[idx]
            contrib = pd.DataFrame({
                "Öznitelik": feat_cols,
                "SHAP Değeri": sv,
                "Değer": X_test.iloc[idx].values
            }).sort_values("SHAP Değeri", key=abs, ascending=True).tail(15)

            colors = ["#ef4444" if v > 0 else "#22c55e" for v in contrib["SHAP Değeri"]]
            fig = go.Figure(go.Bar(
                x=contrib["SHAP Değeri"],
                y=contrib["Öznitelik"],
                orientation="h",
                marker_color=colors,
                text=[f"{v:.3f}" for v in contrib["SHAP Değeri"]],
                textposition="outside"
            ))
            fig.update_layout(
                title=f"Örnek #{idx} — SHAP Waterfall (Kırmızı=Süre ↑, Yeşil=Süre ↓)",
                xaxis_title="SHAP Değeri (dk cinsinden etki)",
                template="plotly_dark", height=480
            )
            st.plotly_chart(fig, use_container_width=True)

    # ─── FOOTER ─────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("""
    <div style='text-align:center; color:#555; font-size:0.78rem; padding:10px 0'>
        🏭 MDF Üretim Süresi Tahmin Sistemi &nbsp;|&nbsp; 
        Makine Öğrenmesi Tabanlı Endüstriyel Analiz &nbsp;|&nbsp;
        XGBoost · LightGBM · Random Forest · SHAP &nbsp;|&nbsp;
        Bitirme Projesi 2024–2025
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()