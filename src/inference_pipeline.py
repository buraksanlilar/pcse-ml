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

# ─── Sabitler (pipeline01 ile senkronize) ─────────────────────────────────────
COMBO  = ["district_name", "crop_name", "variety_name", "year"]
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
    "Nevşehir, Avanos":                         {"latitude": 38.715, "longitude": 34.845, "soil_type": "Fine Sand"},
    "Aksaray, Merkez":                          {"latitude": 38.368, "longitude": 34.032, "soil_type": "Loamy Fine Sand"},
    "Antalya, Manavgat":                        {"latitude": 36.785, "longitude": 31.443, "soil_type": "Very Loamy Fine Sand"},
    "Sakarya, Karasu":                          {"latitude": 41.102, "longitude": 30.705, "soil_type": "Fine Sandy Loam"},
    "Iğdır, Aralık":                            {"latitude": 39.865, "longitude": 44.512, "soil_type": "Coarse"},
    "Konya, Karapınar":                         {"latitude": 37.715, "longitude": 33.550, "soil_type": "Medium"},
    "Ankara, Polatlı":                          {"latitude": 39.575, "longitude": 32.145, "soil_type": "Medium Fine"},
    "Edirne, Uzunköprü":                        {"latitude": 41.270, "longitude": 26.685, "soil_type": "Fine Vertisols"},
    "Adana, Karataş":                           {"latitude": 36.575, "longitude": 35.375, "soil_type": "Very Fine Heavy Clay"},
    "Trabzon, Köprübaşı (Ağaçbaşı)":            {"latitude": 40.758, "longitude": 40.125, "soil_type": "Peat Organic"},
    "Şanlıurfa, Akçakale":                      {"latitude": 36.711, "longitude": 38.948, "soil_type": "Silt Loam Clay Loam"},
    "İzmir, Menemen":                           {"latitude": 38.605, "longitude": 27.065, "soil_type": "Loam Alluvial"},
    "Bursa, Karacabey":                         {"latitude": 40.215, "longitude": 28.355, "soil_type": "Medium Fine"},
    "Muş, Merkez":                              {"latitude": 38.745, "longitude": 41.505, "soil_type": "Very Fine Heavy Clay"},
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
            dvs=0.5,
            lai=3.5,
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
        cm = MODEL_DIR / "category_maps.pkl"
        vm = MODEL_DIR / "variety_map.pkl"

        if not mp.exists():
            mp = MODEL_DIR / "lgbm_yield_final.pkl"

        self.model         = joblib.load(mp)
        self.feature_cols  = joblib.load(fp)
        self.model_path    = mp
        self.category_maps = joblib.load(cm) if cm.exists() else {}
        self.variety_map   = (
            joblib.load(vm) if vm.exists()
            else self.category_maps.get("variety_name", {})
        )

        # FIX: feature_cols içinde rolling varsa uyar (02_model_lightgbm ile tutarsızlık)
        rolling_in_features = [c for c in self.feature_cols if c.startswith("roll")]
        if rolling_in_features:
            print(
                f"[YieldPredictor] UYARI: feature_cols.pkl içinde {len(rolling_in_features)} rolling "
                f"feature var ({rolling_in_features[:3]}...). "
                "02_model_lightgbm'i tekrar çalıştırıp yeni feature_cols.pkl kaydedin."
            )

        if not self.category_maps:
            print("[YieldPredictor] Uyarı: category_maps.pkl bulunamadı, varsayılan sabit map kullanılacak.")

        print(f"[YieldPredictor] Model yüklendi  : {mp.name}")
        print(f"[YieldPredictor] Feature sayısı  : {len(self.feature_cols)}")
        print(f"[YieldPredictor] Variety map girdisi: {len(self.variety_map)}")

    # ── Dahili yardımcılar ────────────────────────────────────────────────────
    @staticmethod
    def _dvs_donem(dvs):
        if pd.isna(dvs) or dvs is None: return "pre_sowing"
        elif dvs < 0:  return "germination"
        elif dvs < 1:  return "vegetative"
        elif dvs < 2:  return "reproductive"
        else:          return "maturity"

    def _get_cat_map(self, key: str, fallback_values=None):
        if key in self.category_maps:
            return self.category_maps[key]
        if key == "variety_name" and self.variety_map:
            return self.variety_map
        fallback_values = fallback_values or []
        return {v: i for i, v in enumerate(sorted(set(fallback_values)))}

    @staticmethod
    def _standardize_input_df(data: pd.DataFrame) -> pd.DataFrame:
        alias = {
            "district":           "district_name",
            "crop":               "crop_name",
            "air_temp_mean":      "AIR_TEMP_mean",
            "air_temp_min":       "AIR_TEMP_min",
            "air_temp_max":       "AIR_TEMP_max",
            "air_humidity_mean":  "AIR_HUMIDITY_mean",
            "air_humidity_min":   "AIR_HUMIDITY_min",
            "air_humidity_max":   "AIR_HUMIDITY_max",
            "precip_sum":         "PRECIP_sum",
            "soil_temp_mean":     "SOIL_TEMP_0_7_mean",
            "soil_moisture_mean": "SOIL_MOISTURE_0_7_mean",
            "dvs":                "DVS",
            "lai":                "LAI",
            "tagp":               "TAGP",
            "twso":               "TWSO",
            "twlv":               "TWLV",
            "twst":               "TWST",
            "twrt":               "TWRT",
            "tra":                "TRA",
            "rd":                 "RD",
            "sm":                 "SM",
            "wwlow":              "WWLOW",
            "rftra":              "RFTRA",
            "gdd_cumsum":         "GDD_cumsum",
            "precip_cumsum":      "PRECIP_cumsum",
            "day_of_year":        "day_of_year",
            "month":              "month",
            "week":               "week",
        }
        out = data.copy()
        for old, new in alias.items():
            if old in out.columns and new not in out.columns:
                out[new] = out[old]
        if "district_name" not in out.columns:
            out["district_name"] = "Konya, Karapınar"
        if "crop_name" not in out.columns:
            out["crop_name"] = "wheat"
        if "variety_name" not in out.columns:
            out["variety_name"] = "Unknown"
        return out

    def _build_features_df(self, data: pd.DataFrame) -> pd.DataFrame:
        df = self._standardize_input_df(data)
        df = df.copy()

        # FIX: DVS default 0.5 (vegetative dönem) — nan bırakmak growth_stage'i bozuyor
        defaults = {
            "AIR_TEMP_mean":          20.0,
            "AIR_TEMP_min":           10.0,
            "AIR_TEMP_max":           30.0,
            "AIR_HUMIDITY_mean":      60.0,
            "AIR_HUMIDITY_min":       30.0,
            "AIR_HUMIDITY_max":       90.0,
            "PRECIP_sum":              0.0,
            "SOIL_TEMP_0_7_mean":     15.0,
            "SOIL_MOISTURE_0_7_mean":  0.25,
            "DVS":                     0.5,   # FIX: nan → 0.5 (vegetative)
            "LAI":                     2.0,
            "TAGP":                    0.0,
            "TWSO":                    0.0,
            "TWLV":                    0.0,
            "TWST":                    0.0,
            "TWRT":                    0.0,
            "TRA":                     1.0,
            "RD":                     30.0,
            "SM":                      0.25,
            "WWLOW":                  80.0,
            "RFTRA":                   0.9,
            "GDD_cumsum":            500.0,
            "PRECIP_cumsum":         100.0,
            "day_of_year":           180,
            "month":                   6,
            "week":                   26,
        }

        for col, default in defaults.items():
            if col not in df.columns:
                df[col] = default
            else:
                df[col] = df[col].fillna(default)

        # Bölge meta bilgisi
        meta_df = pd.DataFrame.from_dict(BOLGE_META, orient="index")
        meta_df.index.name = "district_name"
        meta_df = meta_df.reset_index()
        df = df.merge(meta_df, on="district_name", how="left", suffixes=("", "_meta"))
        df["latitude"]  = df["latitude"].fillna(38.0)
        df["longitude"] = df["longitude"].fillna(34.0)
        df["soil_type"] = df["soil_type"].fillna("Medium")

        # Türetilmiş feature'lar
        df["tbase"]      = df["crop_name"].map(TBASE).fillna(5)
        df["GDD_daily"]  = (df["AIR_TEMP_mean"] - df["tbase"]).clip(lower=0)
        df["TEMP_range"] = df["AIR_TEMP_max"] - df["AIR_TEMP_min"]

        if "growth_stage" not in df.columns:
            df["growth_stage"] = df["DVS"].apply(self._dvs_donem)

        # FIX: Rolling feature'lar artık modelde yok (02_model_lightgbm'de DROP_COLS'a alındı).
        # feature_cols içinde rolling varsa eski pkl kullanılıyor demektir — base kolondan doldur.
        required_roll = [c for c in self.feature_cols if c.startswith("roll")]
        if required_roll:
            for rc in required_roll:
                base_col = rc.split("_roll")[0]
                if base_col in df.columns:
                    df[rc] = df[base_col]
                else:
                    df[rc] = 0.0

        # Encoding
        crop_map     = self._get_cat_map("crop_name",     TBASE.keys())
        district_map = self._get_cat_map("district_name", BOLGE_META.keys())
        soil_map     = self._get_cat_map("soil_type",     SOIL_TYPE_MAP.keys())
        growth_map   = self._get_cat_map("growth_stage",  ["germination", "maturity", "pre_sowing", "reproductive", "vegetative"])

        df["crop_name_enc"]      = df["crop_name"].astype(str).map(crop_map).fillna(-1).astype(int)
        df["variety_name_enc"]   = df["variety_name"].astype(str).map(self.variety_map).fillna(-1).astype(int)
        df["district_name_enc"]  = df["district_name"].astype(str).map(district_map).fillna(-1).astype(int)
        df["soil_type_enc"]      = df["soil_type"].astype(str).map(soil_map).fillna(-1).astype(int)
        df["growth_stage_enc"]   = df["growth_stage"].astype(str).map(growth_map).fillna(-1).astype(int)

        # FIX: year feature listesinde varsa 0 yerine makul default (2024) kullan
        if "year" in self.feature_cols and "year" not in df.columns:
            df["year"] = 2024

        # Eksik kalan tüm feature'ları 0 ile doldur
        for c in self.feature_cols:
            if c not in df.columns:
                df[c] = 0.0

        return df[self.feature_cols]

    def _build_row(self, row: dict) -> pd.DataFrame:
        """Ham girdi dict'ini model feature vektörüne dönüştür."""
        return self._build_features_df(pd.DataFrame([row]))

    # ── Tek tahmin ────────────────────────────────────────────────────────────
    def predict_single(self, district: str, crop: str, **kwargs) -> dict:
        row  = {"district_name": district, "crop_name": crop, **kwargs}
        X    = self._build_row(row)
        pred = float(self.model.predict(X)[0])
        pred = max(pred, 0.0)

        result = {
            "district":     district,
            "crop":         crop,
            "variety":      kwargs.get("variety_name", "Unknown"),
            "twso_pred":    round(pred, 1),
            "growth_stage": self._dvs_donem(kwargs.get("dvs", 0.5)),
            "input_summary": {
                "air_temp_mean":  kwargs.get("air_temp_mean"),
                "precip_sum":     kwargs.get("precip_sum"),
                "dvs":            kwargs.get("dvs"),
                "gdd_cumsum":     kwargs.get("gdd_cumsum"),
            },
        }
        return result

    # ── Batch tahmin (CSV / DataFrame) ────────────────────────────────────────
    def predict_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        X_all = self._build_features_df(data)
        preds = self.model.predict(X_all)
        preds = np.clip(preds, a_min=0.0, a_max=None)
        out   = data.copy()
        out["twso_pred"] = np.round(preds, 1)
        return out

    # ── Güven aralığı (bootstrap) ─────────────────────────────────────────────
    def predict_with_uncertainty(self, district: str, crop: str,
                                  n_bootstrap: int = 50, **kwargs) -> dict:
        """
        Bootstrap ile yaklaşık güven aralığı.
        Hızlı tahmin için n_bootstrap=30 yeterli.
        """
        row   = {"district_name": district, "crop_name": crop, **kwargs}
        X     = self._build_row(row)
        preds = []

        try:
            booster = self.model.booster_
            n_trees = booster.num_trees()
            step    = max(1, n_trees // n_bootstrap)
            for i in range(0, n_trees, step):
                p = booster.predict(X.values, num_iteration=i + 1)
                preds.append(float(p[0]))
        except Exception:
            base_pred = max(float(self.model.predict(X)[0]), 0.0)
            sigma     = max(abs(base_pred) * 0.05, 1e-6)
            noise     = np.random.normal(0, sigma, n_bootstrap)
            preds     = (base_pred + noise).tolist()

        preds  = np.clip(np.array(preds, dtype=float), a_min=0.0, a_max=None)
        result = {
            "district":  district,
            "crop":      crop,
            "twso_pred": round(float(np.mean(preds)), 1),
            "ci_lower":  round(float(np.percentile(preds, 10)), 1),
            "ci_upper":  round(float(np.percentile(preds, 90)), 1),
            "std":       round(float(np.std(preds)), 1),
        }
        return result


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════
def interactive_demo(predictor: YieldPredictor):
    print("\n" + "=" * 55)
    print("  PCSE Verim Tahmin — İnteraktif Demo")
    print("=" * 55)

    districts = sorted(BOLGE_META.keys())
    crops     = sorted(TBASE.keys())

    print("\nMevcut bölgeler:")
    for i, d in enumerate(districts, 1):
        print(f"  {i:2d}. {d}")
    try:
        d_idx    = int(input("\nBölge numarası seç: ")) - 1
        district = districts[d_idx]
    except Exception:
        district = "Konya, Karapınar"
        print(f"Geçersiz — varsayılan: {district}")

    print("\nMevcut ürünler:")
    for i, c in enumerate(crops, 1):
        print(f"  {i:2d}. {c}")
    try:
        c_idx = int(input("\nÜrün numarası seç: ")) - 1
        crop  = crops[c_idx]
    except Exception:
        crop = "wheat"
        print(f"Geçersiz — varsayılan: {crop}")

    def ask(prompt, default):
        val = input(f"{prompt} [{default}]: ").strip()
        try:    return float(val)
        except: return default

    print("\nMeteoroji & durum (Enter = varsayılan):")
    params = {
        "air_temp_mean":     ask("Ort. hava sıcaklığı (°C)",  18.0),
        "air_temp_min":      ask("Min sıcaklık (°C)",          8.0),
        "air_temp_max":      ask("Max sıcaklık (°C)",         28.0),
        "air_humidity_mean": ask("Ort. nem (%)",              55.0),
        "precip_sum":        ask("Günlük yağış (mm)",          2.0),
        "soil_moisture_mean":ask("Toprak nemi (0-1)",          0.28),
        "dvs":               ask("DVS (0-2)",                  0.5),   # FIX: default 0.5
        "gdd_cumsum":        ask("Kümülatif GDD",           1000.0),
        "precip_cumsum":     ask("Kümülatif yağış (mm)",     150.0),
        "day_of_year":       ask("Yılın günü (1-365)",        150.0),
    }

    print("\n⏳ Tahmin yapılıyor...")
    result = predictor.predict_single(district, crop, **params)
    unc    = predictor.predict_with_uncertainty(district, crop, **params)

    print("\n" + "=" * 55)
    print("  SONUÇ")
    print("=" * 55)
    print(f"  Bölge         : {result['district']}")
    print(f"  Ürün          : {result['crop']}")
    print(f"  Büyüme dönemi : {result['growth_stage']}")
    print(f"  Tahmin verim  : {result['twso_pred']:>8.1f}  kg/ha")
    print(f"  %80 güven     : {unc['ci_lower']:.1f} — {unc['ci_upper']:.1f}")
    print("=" * 55)
    return result


def batch_from_csv(predictor: YieldPredictor, input_path: str, output_path: str):
    print(f"\nBatch tahmin: {input_path}")
    data = pd.read_csv(input_path)
    out  = predictor.predict_batch(data)
    out.to_csv(output_path, index=False)
    print(f"Tahminler kaydedildi: {output_path}")
    print(out[["district_name", "crop_name", "twso_pred"]].head(10).to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PCSE Yield Inference Pipeline")
    parser.add_argument("--input",  type=str, help="Batch CSV input yolu")
    parser.add_argument("--output", type=str, default="data/tahminler.csv", help="Batch CSV output yolu")
    parser.add_argument("--model",  type=str, help="Model pkl yolu (opsiyonel)")
    args = parser.parse_args()

    predictor = YieldPredictor(model_path=args.model)

    if args.input:
        batch_from_csv(predictor, args.input, args.output)
    else:
        interactive_demo(predictor)