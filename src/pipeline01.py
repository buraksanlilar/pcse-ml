"""
PCSE Yield Prediction Pipeline - Adım 1
Temizleme + Feature Engineering + ML-hazır dataset üretimi

Proje yapısı:
  pcse-ml/
  ├── data/
  │   ├── final_hourly_pcse_dataset_all_crops.csv  ← ham veri
  │   ├── ml_dataset_gunluk.parquet                ← bu script üretir
  │   ├── kombinasyon_flags.csv                    ← bu script üretir
  │   └── cikartilan_kombinasyonlar.csv            ← bu script üretir
  ├── src/
  │   └── pipeline_01.py                           ← bu dosya
  └── notebooks/

Çalıştır (pcse-ml/ klasöründen):
  python src/pipeline_01.py
"""

import pandas as pd
import dask.dataframe as dd
import numpy as np
from pathlib import Path

# ─── Yollar ───────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent   # pcse-ml/
DATA_DIR   = BASE_DIR / "data"
INPUT_FILE = DATA_DIR / "final_hourly_pcse_dataset_all_crops.csv"
DATA_DIR.mkdir(exist_ok=True)

# ─── Sabitler ─────────────────────────────────────────────────────────────────
TWSO_SIFIR_ESIK = 0.0   # eşit veya altı → kombinasyon tamamen çıkarılır

PCSE_COLS = ["DVS", "LAI", "TAGP", "TWSO", "TWLV", "TWST",
             "TWRT", "TRA", "RD", "SM", "WWLOW", "RFTRA"]

TBASE = {
    "barley": 0,      "wheat": 0,        "rapeseed": 0,
    "maize": 10,      "sorghum": 10,     "sunflower": 6,
    "soybean": 6,     "rice": 8,         "potato": 2,
    "fababean": 0,    "chickpea": 0,     "cowpea": 10,
    "mungbean": 10,   "millet": 10,      "pigeonpea": 10,
    "seed_onion": 2,  "sweetpotato": 10, "sugarbeet": 3,
    "cassava": 10,    "cotton": 15,      "groundnut": 9,
    "sugarcane": 12,  "tobacco": 13,
}

BOLGE_META = pd.DataFrame([
    {"district_name": "Nevşehir, Avanos",
        "latitude": 38.715, "longitude": 34.845, "soil_type": "Fine Sand"},
    {"district_name": "Aksaray, Merkez",
        "latitude": 38.368, "longitude": 34.032, "soil_type": "Loamy Fine Sand"},
    {"district_name": "Antalya, Manavgat",
        "latitude": 36.785, "longitude": 31.443, "soil_type": "Very Loamy Fine Sand"},
    {"district_name": "Sakarya, Karasu",
        "latitude": 41.102, "longitude": 30.705, "soil_type": "Fine Sandy Loam"},
    {"district_name": "Iğdır, Aralık",
        "latitude": 39.865, "longitude": 44.512, "soil_type": "Coarse"},
    {"district_name": "Konya, Karapınar",
        "latitude": 37.715, "longitude": 33.550, "soil_type": "Medium"},
    {"district_name": "Ankara, Polatlı",
        "latitude": 39.575, "longitude": 32.145, "soil_type": "Medium Fine"},
    {"district_name": "Edirne, Uzunköprü",
        "latitude": 41.270, "longitude": 26.685, "soil_type": "Fine Vertisols"},
    {"district_name": "Adana, Karataş",
        "latitude": 36.575, "longitude": 35.375, "soil_type": "Very Fine Heavy Clay"},
    {"district_name": "Trabzon, Köprübaşı (Ağaçbaşı)",
        "latitude": 40.758, "longitude": 40.125, "soil_type": "Peat Organic"},
    {"district_name": "Şanlıurfa, Akçakale",
        "latitude": 36.711, "longitude": 38.948, "soil_type": "Silt Loam Clay Loam"},
    {"district_name": "İzmir, Menemen",
        "latitude": 38.605, "longitude": 27.065, "soil_type": "Loam Alluvial"},
    {"district_name": "Bursa, Karacabey",
        "latitude": 40.215, "longitude": 28.355, "soil_type": "Medium Fine"},
    {"district_name": "Muş, Merkez",
        "latitude": 38.745, "longitude": 41.505, "soil_type": "Very Fine Heavy Clay"},
    {"district_name": "Kayseri, Develi (Sultan Sazlığı çevresi)",
        "latitude": 38.355, "longitude": 35.215, "soil_type": "High Retention Clay Organic"},
])

COMBO = ["district_name", "crop_name", "variety_name"]


# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("PCSE Pipeline - Adım 1")
print("=" * 60)


# ─── [1] Yükle ────────────────────────────────────────────────────────────────
print("\n[1/9] Veri yükleniyor...")
print(f"      {INPUT_FILE}")

df = dd.read_csv(
    INPUT_FILE,
    dtype={
        "DATETIME": "str", "date": "str",
        "district_name": "str", "crop_name": "str", "variety_name": "str"
    }
)


# ─── [2] TWSO = 0 olan kombinasyonları tespit et ve çıkar ─────────────────────
print("\n[2/9] TWSO = 0 olan kombinasyonlar tespit ediliyor...")

twso_check = (
    df[~df["TWSO"].isna()]
    .groupby(COMBO)["TWSO"]
    .last()
    .compute()
    .reset_index()
    .rename(columns={"TWSO": "twso_final"})
)

sifir_kombolar   = twso_check[twso_check["twso_final"] <= TWSO_SIFIR_ESIK]
gecerli_kombolar = twso_check[twso_check["twso_final"] >  TWSO_SIFIR_ESIK].copy()

print(f"      Toplam kombinasyon      : {len(twso_check)}")
print(f"      TWSO = 0 (çıkarılacak)  : {len(sifir_kombolar)}")
print(f"      Geçerli kombinasyon     : {len(gecerli_kombolar)}")
print(f"\n      Çıkarılan ürünler (kombinasyon sayısı):")
print(sifir_kombolar.groupby("crop_name")["district_name"]
      .count().sort_values(ascending=False).to_string())

sifir_kombolar.to_csv(DATA_DIR / "cikartilan_kombinasyonlar.csv", index=False)

# Dask için filtre anahtarı
gecerli_kombolar["_key"] = (
    gecerli_kombolar["district_name"] + "||" +
    gecerli_kombolar["crop_name"]     + "||" +
    gecerli_kombolar["variety_name"]
)
gecerli_keyler = set(gecerli_kombolar["_key"])

df["_key"] = (
    df["district_name"] + "||" +
    df["crop_name"]     + "||" +
    df["variety_name"]
)
df = df[df["_key"].isin(gecerli_keyler)]


# ─── [3] 31 Aralık NaN satırlarını temizle ────────────────────────────────────
print("\n[3/9] Başlangıç NaN satırları temizleniyor...")
df = df[~df["SM"].isna()]


# ─── [4] Tip dönüşümü ─────────────────────────────────────────────────────────
print("\n[4/9] Tarih tipi dönüştürülüyor...")
df["date"] = dd.to_datetime(df["date"])


# ─── [5] Günlük aggregate ─────────────────────────────────────────────────────
print("\n[5/9] Günlük aggregate hesaplanıyor (~2-4 dk)...")

meteo_agg = {
    "AIR_TEMP":          ["mean", "min", "max"],
    "AIR_HUMIDITY":      ["mean", "min", "max"],
    "PRECIP":            ["sum"],
    "SOIL_TEMP_0_7":     ["mean"],
    "SOIL_MOISTURE_0_7": ["mean"],
}
pcse_agg  = {col: "first" for col in PCSE_COLS}
GROUP_KEYS = ["date", "district_name", "crop_name", "variety_name"]

gunluk = (
    df.groupby(GROUP_KEYS)
      .agg({**meteo_agg, **pcse_agg})
      .reset_index()
      .compute()
)

# Sütun isimlerini düzleştir
gunluk.columns = [
    "_".join(c).strip("_") if isinstance(c, tuple) else c
    for c in gunluk.columns
]
for col in PCSE_COLS:
    if f"{col}_first" in gunluk.columns:
        gunluk.rename(columns={f"{col}_first": col}, inplace=True)

gunluk.drop(columns=["_key"], errors="ignore", inplace=True)
print(f"      Günlük satır sayısı: {len(gunluk):,}")


# ─── [6] Feature Engineering ──────────────────────────────────────────────────
print("\n[6/9] Feature engineering...")

gunluk = gunluk.sort_values(GROUP_KEYS).reset_index(drop=True)

# GDD
gunluk["tbase"]     = gunluk["crop_name"].map(TBASE).fillna(5)
gunluk["GDD_daily"] = (gunluk["AIR_TEMP_mean"] - gunluk["tbase"]).clip(lower=0)

# Kümülatif değerler
gunluk["GDD_cumsum"]    = gunluk.groupby(COMBO)["GDD_daily"].cumsum()
gunluk["PRECIP_cumsum"] = gunluk.groupby(COMBO)["PRECIP_sum"].cumsum()

# Rolling ortalamalar
for col in ["AIR_TEMP_mean", "AIR_HUMIDITY_mean", "SOIL_MOISTURE_0_7_mean"]:
    for w in [7, 30]:
        gunluk[f"{col}_roll{w}"] = (
            gunluk.groupby(COMBO)[col]
            .transform(lambda x: x.rolling(w, min_periods=1).mean())
        )

# Sıcaklık aralığı
gunluk["TEMP_range"] = gunluk["AIR_TEMP_max"] - gunluk["AIR_TEMP_min"]

# Takvim
gunluk["day_of_year"] = gunluk["date"].dt.dayofyear
gunluk["month"]       = gunluk["date"].dt.month
gunluk["week"]        = gunluk["date"].dt.isocalendar().week.astype(int)

# DVS büyüme dönemi
def dvs_donem(dvs):
    if pd.isna(dvs):  return "pre_sowing"
    elif dvs < 0:     return "germination"
    elif dvs < 1:     return "vegetative"
    elif dvs < 2:     return "reproductive"
    else:             return "maturity"

gunluk["growth_stage"] = gunluk["DVS"].apply(dvs_donem)
print("      Tüm feature'lar hesaplandı.")


# ─── [7] Target değişkeni ekle ────────────────────────────────────────────────
print("\n[7/9] Target değişkeni ekleniyor...")

gunluk = gunluk.merge(
    gecerli_kombolar[COMBO + ["twso_final"]],
    on=COMBO, how="left"
)

print(f"      twso_final istatistikleri:")
print(gunluk["twso_final"].describe().round(1).to_string())


# ─── [8] Bölge metadata + encoding ───────────────────────────────────────────
print("\n[8/9] Bölge metadata ve encoding...")

gunluk = gunluk.merge(BOLGE_META, on="district_name", how="left")

for col in ["crop_name", "variety_name", "district_name", "soil_type", "growth_stage"]:
    gunluk[f"{col}_enc"] = pd.Categorical(gunluk[col]).codes


# ─── [9] Kaydet ───────────────────────────────────────────────────────────────
print("\n[9/9] Kaydediliyor...")

out_parquet = DATA_DIR / "ml_dataset_gunluk.parquet"
out_flags   = DATA_DIR / "kombinasyon_flags.csv"

gunluk.to_parquet(out_parquet, index=False)

flag_tablo = (
    gunluk.groupby(COMBO + ["twso_final"])
    .agg(gun_sayisi=("date", "count"))
    .reset_index()
    .sort_values("twso_final", ascending=False)
)
flag_tablo.to_csv(out_flags, index=False)

print(f"      {out_parquet}  ({out_parquet.stat().st_size / 1e6:.1f} MB)")
print(f"      {out_flags}")


# ─── Özet ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ÖZET")
print("=" * 60)
print(f"Toplam satır          : {len(gunluk):,}")
print(f"Toplam sütun          : {len(gunluk.columns)}")
print(f"Kombinasyon sayısı    : {gunluk.groupby(COMBO).ngroups}")
print(f"Tarih aralığı         : {gunluk['date'].min().date()} → {gunluk['date'].max().date()}")
print(f"\nÜrün bazında ortalama verim (twso_final):")
print(gunluk.groupby("crop_name")["twso_final"]
      .mean().round(1).sort_values(ascending=False).to_string())
print("\nHazır! Sonraki: notebooks/02_model_lightgbm.ipynb")