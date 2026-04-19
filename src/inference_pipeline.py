"""
PCSE Yield Prediction — Inference Pipeline
Yeni veri → twso_final tahmini

Kullanım:
  # Tek tahmin (interaktif)
  python src/04_inference_pipeline.py

  # CSV batch tahmini
  python src/04_inference_pipeline.py --input data/yeni_veri.csv --output data/tahminler.csv

  # Python'dan import
  from src.inference_pipeline import YieldPredictor
  predictor = YieldPredictor()
  result = predictor.predict_single(district="Konya, Karapınar", crop="wheat", ...)
"""

import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─── Yollar ───────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent.parent
DATA_DIR  = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

# ─── Sabitler (pipeline_01 ile senkronize) ────────────────────────────────────
COMBO  = ["district_name", "crop_name", "variety_name"]
TARGET = "twso_final"

TBASE = {
    "barley": 0, "wheat": 0, "rapeseed": 0,
    "maize": 10, "sorghum": 10, "sunflower": 6,
    "soybean": 6, "rice": 8, "potato": 2,
    "fababean": 0, "chickpea": 0, "cowpea": 10,
    "mungbean": 10, "millet": 10, "pigeonpea": 10,
    "seed_onion": 2, "sweetpotato": 10, "sugarbeet": 3,
    "cassava": 10, "cotton": 15, "groundnut": 9,
    "sugarcane": 12, "tobacco": 13,
}

BOLGE_META = {
    "Nevşehir, Avanos":                    {"latitude": 38.715, "longitude": 34.845, "soil_type": "Fine Sand"},
    "Aksaray, Merkez":                     {"latitude": 38.368, "longitude": 34.032, "soil_type": "Loamy Fine Sand"},
    "Antalya, Manavgat":                   {"latitude": 36.785, "longitude": 31.443, "soil_type": "Very Loamy Fine Sand"},
    "Sakarya, Karasu":                     {"latitude": 41.102, "longitude": 30.705, "soil_type": "Fine Sandy Loam"},
    "Iğdır, Aralık":                       {"latitude": 39.865, "longitude": 44.512, "soil_type": "Coarse"},
    "Konya, Karapınar":                    {"latitude": 37.715, "longitude": 33.550, "soil_type": "Medium"},
    "Ankara, Polatlı":                     {"latitude": 39.575, "longitude": 32.145, "soil_type": "Medium Fine"},
    "Edirne, Uzunköprü":                   {"latitude": 41.270, "longitude": 26.685, "soil_type": "Fine Vertisols"},
    "Adana, Karataş":                      {"latitude": 36.575, "longitude": 35.375, "soil_type": "Very Fine Heavy Clay"},
    "Trabzon, Köprübaşı (Ağaçbaşı)":       {"latitude": 40.758, "longitude": 40.125, "soil_type": "Peat Organic"},
    "Şanlıurfa, Akçakale":                 {"latitude": 36.711, "longitude": 38.948, "soil_type": "Silt Loam Clay Loam"},
    "İzmir, Menemen":                      {"latitude": 38.605, "longitude": 27.065, "soil_type": "Loam Alluvial"},
    "Bursa, Karacabey":                    {"latitude": 40.215, "longitude": 28.355, "soil_type": "Medium Fine"},
    "Muş, Merkez":                         {"latitude": 38.745, "longitude": 41.505, "soil_type": "Very Fine Heavy Clay"},
    "Kayseri, Develi (Sultan Sazlığı çevresi)": {"latitude": 38.355, "longitude": 35.215, "soil_type": "High Retention Clay Organic"},
}

DVS_STAGE_MAP = {
    "pre_sowing":   -1,
    "germination":  -0.5,
    "vegetative":    0.5,
    "reproductive":  1.5,
    "maturity":      2.0,
}

SOIL_TYPE_MAP = {s: i for i, s in enumerate(sorted({v["soil_type"] for v in BOLGE_META.values()}))}


# ══════════════════════════════════════════════════════════════════════════════
class YieldPredictor:
    """
    PCSE verim tahmin sınıfı.

    Örnek:
        predictor = YieldPredictor()
        sonuc = predictor.predict_single(
            district="Konya, Karapınar",
            crop="wheat",
            variety="Winter Wheat",
            air_temp_mean=18.5,
            air_temp_min=10.0,
            air_temp_max=27.0,
            air_humidity_mean=55.0,
            air_humidity_min=30.0,
            air_humidity_max=80.0,
            precip_sum=2.5,
            soil_temp_mean=15.0,
            soil_moisture_mean=0.28,
            dvs=1.2,
            lai=3.5,
            tagp=8000.0,
            twlv=1200.0,
            twst=600.0,
            twrt=400.0,
            tra=2.1,
            rd=50.0,
            sm=0.28,
            wwlow=100.0,
            rftra=0.9,
            gdd_cumsum=1200.0,
            precip_cumsum=180.0,
            day_of_year=150,
        )
        print(sonuc)
    """

    def __init__(self, model_path: str = None, features_path: str = None):
        mp = Path(model_path) if model_path else MODEL_DIR / "best_model.pkl"
        fp = Path(features_path) if features_path else MODEL_DIR / "feature_cols.pkl"

        if not mp.exists():
            mp = MODEL_DIR / "lgbm_yield_final.pkl"

        self.model        = joblib.load(mp)
        self.feature_cols = joblib.load(fp)
        self.model_path   = mp
        print(f"[YieldPredictor] Model yüklendi: {mp.name}")
        print(f"[YieldPredictor] Feature sayısı : {len(self.feature_cols)}")

    # ── Dahili yardımcılar ────────────────────────────────────────────────────
    @staticmethod
    def _dvs_donem(dvs):
        if pd.isna(dvs) or dvs is None: return "pre_sowing"
        elif dvs < 0:   return "germination"
        elif dvs < 1:   return "vegetative"
        elif dvs < 2:   return "reproductive"
        else:           return "maturity"

    @staticmethod
    def _encode_categorical(value, mapping):
        return mapping.get(value, -1)

    def _build_row(self, row: dict) -> pd.DataFrame:
        """Ham girdi dict'ini model feature vektörüne dönüştür."""
        district  = row["district_name"]
        crop      = row["crop_name"]
        variety   = row.get("variety_name", "Unknown")
        meta      = BOLGE_META.get(district, {})
        tbase     = TBASE.get(crop, 5)
        dvs       = row.get("dvs", row.get("DVS", np.nan))
        growth_st = self._dvs_donem(dvs)

        # Tüm ürün/bölge/variety encoding'leri için
        # eğitim verisiyle senkronize sabit kodlar kullan
        all_crops      = sorted(TBASE.keys())
        all_districts  = sorted(BOLGE_META.keys())
        all_soils      = sorted(SOIL_TYPE_MAP.keys())
        all_stages     = ["germination", "maturity", "pre_sowing", "reproductive", "vegetative"]

        feat = {
            # Meteoroloji
            "AIR_TEMP_mean":           row.get("air_temp_mean", 20.0),
            "AIR_TEMP_min":            row.get("air_temp_min", 10.0),
            "AIR_TEMP_max":            row.get("air_temp_max", 30.0),
            "AIR_HUMIDITY_mean":       row.get("air_humidity_mean", 60.0),
            "AIR_HUMIDITY_min":        row.get("air_humidity_min", 30.0),
            "AIR_HUMIDITY_max":        row.get("air_humidity_max", 90.0),
            "PRECIP_sum":              row.get("precip_sum", 0.0),
            "SOIL_TEMP_0_7_mean":      row.get("soil_temp_mean", 15.0),
            "SOIL_MOISTURE_0_7_mean":  row.get("soil_moisture_mean", 0.25),
            # PCSE durumu
            "DVS":   dvs,
            "LAI":   row.get("lai",   row.get("LAI",   np.nan)),
            "TAGP":  row.get("tagp",  row.get("TAGP",  np.nan)),
            "TWSO":  row.get("twso",  row.get("TWSO",  np.nan)),
            "TWLV":  row.get("twlv",  row.get("TWLV",  np.nan)),
            "TWST":  row.get("twst",  row.get("TWST",  np.nan)),
            "TWRT":  row.get("twrt",  row.get("TWRT",  np.nan)),
            "TRA":   row.get("tra",   row.get("TRA",   np.nan)),
            "RD":    row.get("rd",    row.get("RD",    np.nan)),
            "SM":    row.get("sm",    row.get("SM",    np.nan)),
            "WWLOW": row.get("wwlow", row.get("WWLOW", np.nan)),
            "RFTRA": row.get("rftra", row.get("RFTRA", np.nan)),
            # Feature engineering
            "GDD_daily":    max(row.get("air_temp_mean", 20.0) - tbase, 0),
            "GDD_cumsum":   row.get("gdd_cumsum",   500.0),
            "PRECIP_cumsum":row.get("precip_cumsum",100.0),
            "TEMP_range":   row.get("air_temp_max", 30.0) - row.get("air_temp_min", 10.0),
            "day_of_year":  row.get("day_of_year",  180),
            "month":        row.get("month",        6),
            "week":         row.get("week",         26),
            # Rolling (girdi yoksa güncel değeri kullan)
            "AIR_TEMP_mean_roll7":          row.get("air_temp_mean", 20.0),
            "AIR_TEMP_mean_roll30":         row.get("air_temp_mean", 20.0),
            "AIR_HUMIDITY_mean_roll7":      row.get("air_humidity_mean", 60.0),
            "AIR_HUMIDITY_mean_roll30":     row.get("air_humidity_mean", 60.0),
            "SOIL_MOISTURE_0_7_mean_roll7": row.get("soil_moisture_mean", 0.25),
            "SOIL_MOISTURE_0_7_mean_roll30":row.get("soil_moisture_mean", 0.25),
            # Coğrafya
            "latitude":  meta.get("latitude",  38.0),
            "longitude": meta.get("longitude", 34.0),
            # Categorical encoding
            "crop_name_enc":      all_crops.index(crop) if crop in all_crops else -1,
            "variety_name_enc":   hash(variety) % 1000,
            "district_name_enc":  all_districts.index(district) if district in all_districts else -1,
            "soil_type_enc":      all_soils.index(meta.get("soil_type","Medium")) if meta.get("soil_type","Medium") in all_soils else -1,
            "growth_stage_enc":   all_stages.index(growth_st) if growth_st in all_stages else -1,
        }

        df_row = pd.DataFrame([feat])
        # Eksik feature'ları 0 ile doldur, fazlalıkları at
        for c in self.feature_cols:
            if c not in df_row.columns:
                df_row[c] = 0.0
        return df_row[self.feature_cols]

    # ── Tek tahmin ────────────────────────────────────────────────────────────
    def predict_single(self, district: str, crop: str, **kwargs) -> dict:
        row = {"district_name": district, "crop_name": crop, **kwargs}
        X   = self._build_row(row)
        pred = float(self.model.predict(X)[0])
        pred = max(pred, 0.0)

        result = {
            "district":    district,
            "crop":        crop,
            "variety":     kwargs.get("variety_name", "Unknown"),
            "twso_pred":   round(pred, 1),
            "growth_stage": self._dvs_donem(kwargs.get("dvs", np.nan)),
            "input_summary": {
                "air_temp_mean":     kwargs.get("air_temp_mean"),
                "precip_sum":        kwargs.get("precip_sum"),
                "dvs":               kwargs.get("dvs"),
                "gdd_cumsum":        kwargs.get("gdd_cumsum"),
            }
        }
        return result

    # ── Batch tahmin (CSV / DataFrame) ────────────────────────────────────────
    def predict_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        results = []
        for _, row in data.iterrows():
            r = dict(row)
            # Kolon ismini normalleştir
            r.setdefault("district_name", r.get("district", "Konya, Karapınar"))
            r.setdefault("crop_name",     r.get("crop",     "wheat"))
            X    = self._build_row(r)
            pred = max(float(self.model.predict(X)[0]), 0.0)
            results.append(pred)
        out = data.copy()
        out["twso_pred"] = np.round(results, 1)
        return out

    # ── Güven aralığı (quantile forest / LightGBM trick) ─────────────────────
    def predict_with_uncertainty(self, district: str, crop: str,
                                  n_bootstrap: int = 50, **kwargs) -> dict:
        """
        Bootstrap ile yaklaşık güven aralığı.
        Hızlı tahmin için n_bootstrap=30 yeterli.
        """
        row    = {"district_name": district, "crop_name": crop, **kwargs}
        X      = self._build_row(row)
        preds  = []

        # LightGBM: her ağaçtan ayrı tahmin al
        try:
            booster = self.model.booster_
            n_trees = booster.num_trees()
            step    = max(1, n_trees // n_bootstrap)
            for i in range(0, n_trees, step):
                p = booster.predict(X.values, num_iteration=i+1)
                preds.append(float(p[0]))
        except Exception:
            # Diğer modeller için basit noise bootstrap
            base_pred = float(self.model.predict(X)[0])
            noise     = np.random.normal(0, base_pred * 0.05, n_bootstrap)
            preds     = (base_pred + noise).tolist()

        preds  = np.array(preds)
        result = {
            "district":   district,
            "crop":       crop,
            "twso_pred":  round(float(np.mean(preds)), 1),
            "ci_lower":   round(float(np.percentile(preds, 10)), 1),
            "ci_upper":   round(float(np.percentile(preds, 90)), 1),
            "std":        round(float(np.std(preds)), 1),
        }
        return result


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════
def interactive_demo(predictor: YieldPredictor):
    print("\n" + "="*55)
    print("  PCSE Verim Tahmin — İnteraktif Demo")
    print("="*55)

    districts = sorted(BOLGE_META.keys())
    crops     = sorted(TBASE.keys())

    print("\nMevcut bölgeler:")
    for i, d in enumerate(districts, 1):
        print(f"  {i:2d}. {d}")
    try:
        d_idx = int(input("\nBölge numarası seç: ")) - 1
        district = districts[d_idx]
    except Exception:
        district = "Konya, Karapınar"
        print(f"Geçersiz — varsayılan: {district}")

    print("\nMevcut ürünler:")
    for i, c in enumerate(crops, 1):
        print(f"  {i:2d}. {c}")
    try:
        c_idx = int(input("\nÜrün numarası seç: ")) - 1
        crop = crops[c_idx]
    except Exception:
        crop = "wheat"
        print(f"Geçersiz — varsayılan: {crop}")

    def ask(prompt, default):
        val = input(f"{prompt} [{default}]: ").strip()
        try:    return float(val)
        except: return default

    print("\nMeteoroji & durum (Enter = varsayılan):")
    params = {
        "air_temp_mean":    ask("Ort. hava sıcaklığı (°C)", 18.0),
        "air_temp_min":     ask("Min sıcaklık (°C)",        8.0),
        "air_temp_max":     ask("Max sıcaklık (°C)",        28.0),
        "air_humidity_mean":ask("Ort. nem (%)",             55.0),
        "precip_sum":       ask("Günlük yağış (mm)",         2.0),
        "soil_moisture_mean":ask("Toprak nemi (0-1)",        0.28),
        "dvs":              ask("DVS (0-2)",                  1.0),
        "gdd_cumsum":       ask("Kümülatif GDD",           1000.0),
        "precip_cumsum":    ask("Kümülatif yağış (mm)",     150.0),
        "day_of_year":      ask("Yılın günü (1-365)",       150.0),
    }

    print("\n⏳ Tahmin yapılıyor...")
    result = predictor.predict_single(district, crop, **params)
    unc    = predictor.predict_with_uncertainty(district, crop, **params)

    print("\n" + "="*55)
    print("  SONUÇ")
    print("="*55)
    print(f"  Bölge        : {result['district']}")
    print(f"  Ürün         : {result['crop']}")
    print(f"  Büyüme dönemi: {result['growth_stage']}")
    print(f"  Tahmin verim : {result['twso_pred']:>8.1f}  kg/ha")
    print(f"  %80 güven    : {unc['ci_lower']:.1f} — {unc['ci_upper']:.1f}")
    print("="*55)
    return result


def batch_from_csv(predictor: YieldPredictor, input_path: str, output_path: str):
    print(f"\nBatch tahmin: {input_path}")
    data = pd.read_csv(input_path)
    out  = predictor.predict_batch(data)
    out.to_csv(output_path, index=False)
    print(f"Tahminler kaydedildi: {output_path}")
    print(out[["district_name","crop_name","twso_pred"]].head(10).to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PCSE Yield Inference Pipeline")
    parser.add_argument("--input",  type=str, help="Batch CSV input yolu")
    parser.add_argument("--output", type=str, help="Batch CSV output yolu", default="data/tahminler.csv")
    parser.add_argument("--model",  type=str, help="Model pkl yolu (opsiyonel)")
    args = parser.parse_args()

    predictor = YieldPredictor(model_path=args.model)

    if args.input:
        batch_from_csv(predictor, args.input, args.output)
    else:
        interactive_demo(predictor)
