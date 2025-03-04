import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(file_path, target_column=None):
    """
    Veri setini yükler ve ön işleme yapar.
    
    Args:
        file_path (str): Veri seti dosya yolu
        target_column (str): Hedef değişken sütun adı
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    # Veriyi yükle
    data = pd.read_csv(file_path)
    
    # Hedef değişkeni ayır
    if target_column:
        X = data.drop(columns=[target_column])
        y = data[target_column]
    else:
        X = data
        y = None
    
    # Veriyi eğitim ve test setlerine ayır
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Özellik ölçeklendirme
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def handle_missing_values(data, strategy='mean'):
    """
    Eksik değerleri işler.
    
    Args:
        data (pd.DataFrame): Veri seti
        strategy (str): İşleme stratejisi ('mean', 'median', 'mode')
    
    Returns:
        pd.DataFrame: İşlenmiş veri seti
    """
    if strategy == 'mean':
        return data.fillna(data.mean())
    elif strategy == 'median':
        return data.fillna(data.median())
    elif strategy == 'mode':
        return data.fillna(data.mode().iloc[0])
    else:
        raise ValueError("Geçersiz strateji. 'mean', 'median' veya 'mode' kullanın.")

def encode_categorical_features(data, columns):
    """
    Kategorik değişkenleri sayısal değerlere dönüştürür.
    
    Args:
        data (pd.DataFrame): Veri seti
        columns (list): Kategorik sütun isimleri
    
    Returns:
        pd.DataFrame: Dönüştürülmüş veri seti
    """
    return pd.get_dummies(data, columns=columns, prefix=columns) 