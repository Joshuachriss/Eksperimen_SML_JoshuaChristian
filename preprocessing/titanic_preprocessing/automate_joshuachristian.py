import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import json
import joblib
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data(filepath: str) -> pd.DataFrame:
    """Memuat dataset dari file CSV."""
    df = pd.read_csv(filepath)
    logger.info(f"Dataset dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")
    logger.info(f"Kolom: {df.columns.tolist()}")
    return df


def drop_unnecessary_columns(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Menghapus kolom yang tidak relevan untuk pemodelan.
    Kolom dihapus: PassengerId, Name, Ticket.
    """
    existing = [c for c in cols if c in df.columns]
    df = df.drop(columns=existing)
    logger.info(f"Kolom dihapus: {existing}")
    return df


def handle_missing_values(df: pd.DataFrame) -> tuple:
    """
    Menangani missing values:
    - Age  : isi dengan median
    - Fare : isi dengan median
    Mengembalikan (DataFrame, dict berisi nilai imputation).
    """
    params = {}

    age_median = df['Age'].median()
    df['Age'] = df['Age'].fillna(age_median)
    params['age_median'] = age_median
    logger.info(f"Age missing diisi median: {age_median:.2f}")

    fare_median = df['Fare'].median()
    df['Fare'] = df['Fare'].fillna(fare_median)
    params['fare_median'] = fare_median
    logger.info(f"Fare missing diisi median: {fare_median:.2f}")

    remaining = df.isnull().sum().sum()
    logger.info(f"Sisa missing values: {remaining}")
    return df, params


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Menghapus baris duplikat."""
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    logger.info(f"Duplikat dihapus: {removed} baris. Sisa: {len(df)}")
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menambahkan fitur baru:
    - FamilySize = SibSp + Parch + 1
    """
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    logger.info(f"FamilySize dibuat. Nilai unik: {sorted(df['FamilySize'].unique().tolist())}")
    return df


def encode_sex(df: pd.DataFrame) -> tuple:
    """
    Encoding kolom Sex menjadi numerik:
    male -> 1, female -> 0
    """
    sex_mapping = {'male': 1, 'female': 0}
    df['Sex'] = df['Sex'].map(sex_mapping)
    logger.info(f"Encoding Sex: {sex_mapping}")
    return df, sex_mapping

def encode_embarked(df):
    df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
    return df

def handle_outliers_fare(df: pd.DataFrame) -> tuple:
    """
    Menangani outlier Fare menggunakan IQR Clipping.
    Mengembalikan (DataFrame, dict berisi batas clipping).
    """
    Q1 = df['Fare'].quantile(0.25)
    Q3 = df['Fare'].quantile(0.75)
    IQR = Q3 - Q1
    fare_upper = Q3 + 1.5 * IQR
    n_clipped = (df['Fare'] > fare_upper).sum()
    df['Fare'] = df['Fare'].clip(upper=fare_upper)
    params = {'fare_clip_upper': fare_upper}
    logger.info(f"Fare outlier clipping: {n_clipped} nilai di-clip, batas atas={fare_upper:.2f}")
    return df, params


def scale_features(df: pd.DataFrame, scale_cols: list, output_dir: str) -> tuple:
    """
    Standarisasi fitur numerik menggunakan StandardScaler.
    Menyimpan scaler.pkl ke output_dir.
    """
    scaler = StandardScaler()
    df[scale_cols] = scaler.fit_transform(df[scale_cols])

    scaler_path = os.path.join(output_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    logger.info(f"Standarisasi selesai: {scale_cols}")
    logger.info(f"Scaler disimpan: {scaler_path}")
    return df, scaler


def save_preprocessing_params(params: dict, output_dir: str) -> None:
    """
    Menyimpan parameter preprocessing (median, mapping, clip bounds)
    ke file JSON agar bisa digunakan saat inference.
    """
    params_path = os.path.join(output_dir, 'preprocessing_params.json')
    # Konversi numpy types ke Python native
    clean = {k: float(v) if isinstance(v, (np.floating, float)) else v
             for k, v in params.items()}
    with open(params_path, 'w') as f:
        json.dump(clean, f, indent=4)
    logger.info(f"Preprocessing params disimpan: {params_path}")

def preprocess(input_path: str, output_path: str) -> pd.DataFrame:
    """
    Pipeline preprocessing lengkap untuk dataset Titanic.

    Tahapan:
        1. Load data
        2. Drop kolom tidak relevan (PassengerId, Name, Ticket)
        3. Tangani missing values (Age, Fare -> median)
        4. Hapus duplikat
        5. Feature engineering (FamilySize)
        6. Encoding Sex (male=1, female=0) dan embaked
        7. Tangani outlier Fare (IQR clipping)
        8. Standarisasi fitur numerik (Age, Fare, FamilySize)
        9. Simpan hasil + scaler + params

    Args:
        input_path : Path ke file CSV raw
        output_path: Path tujuan file CSV hasil preprocessing

    Returns:
        DataFrame hasil preprocessing
    """
    logger.info("=" * 55)
    logger.info("MEMULAI PIPELINE PREPROCESSING TITANIC")
    logger.info("=" * 55)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    all_params = {}

    # Step 1: Load
    df = load_data(input_path)

    # Step 2: Drop kolom tidak relevan
    df = drop_unnecessary_columns(df, ['PassengerId', 'Name', 'Ticket', 'Cabin'])

    # Step 3: Tangani missing values
    df, mv_params = handle_missing_values(df)
    all_params.update(mv_params)

    # Step 4: Hapus duplikat
    df = remove_duplicates(df)

    # Step 5: Feature engineering
    df = feature_engineering(df)

    # Step 6: Encoding Sex
    df, sex_mapping = encode_sex(df)
    all_params['sex_mapping'] = sex_mapping
    df = encode_embarked(df)

    # Step 7: Tangani outlier Fare
    df, outlier_params = handle_outliers_fare(df)
    all_params.update(outlier_params)

    # Step 8: Standarisasi
    scale_cols = ['Age', 'Fare', 'FamilySize']
    df, scaler = scale_features(df, scale_cols, output_dir)
    all_params['scale_cols'] = scale_cols

    # Step 9: Simpan
    df.to_csv(output_path, index=False)
    logger.info(f"Dataset preprocessing tersimpan: {output_path}")
    save_preprocessing_params(all_params, output_dir)

    logger.info("=" * 55)
    logger.info(f"SELESAI | Shape: {df.shape} | Kolom: {df.columns.tolist()}")
    logger.info("=" * 55)

    return df


if __name__ == "__main__":
    INPUT_PATH = "titanic_raw/titanic.csv"
    OUTPUT_PATH = "preprocessing/titanic_preprocessing/titanic_preprocessing.csv"

    result = preprocess(input_path=INPUT_PATH, output_path=OUTPUT_PATH)
    print(f"\nPreview hasil preprocessing:")
    print(result.head())
    print(f"\nShape: {result.shape}")
    print(f"Missing values: {result.isnull().sum().sum()}")
