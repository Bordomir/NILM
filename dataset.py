import os
from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from mappings import REFIT_CANONICAL_MAPPINGS


# ==========================================
# KROK 1: Pobieranie zbioru z Kaggle
# ==========================================
def download_kaggle_dataset(download_dir: str | None = None) -> str:
    """
    Pobiera zbiór z Kaggle za pomocą kagglehub.dataset_download do folderu
    podanego opcjonalnym parametrem i zwraca ścieżkę do danych.
    """
    if download_dir:
        # Pobranie do wskazanej lokalizacji
        path = kagglehub.dataset_download("kyleahmurphy/uk-electrical-load", output_dir=download_dir)
        return path
    else:
        path = kagglehub.dataset_download("kyleahmurphy/uk-electrical-load")
        return path


# ==========================================
# KROK 2-5: Wczytywanie, czyszczenie i mapowanie plików CSV
# ==========================================
def load_and_process_csv_files(dataset_path: str, replace_aggregate_with_sum: bool = False) -> dict:
    """
    Iteruje po folderze w poszukiwaniu plików .csv, przetwarza je zgodnie z planem:
    - wczytanie za pomocą pandas,
    - wydzielenie kolumny 'Time' jako index,
    - usunięcie kolumny 'Unix',
    - opcjonalna zamiana kolumny 'Aggregate' sumą kolumn ApplianceX,
    - zamiana nazw kolumn ApplianceX według słownika.
    Zwraca słownik przetworzonych ramek danych dla poszczególnych domostw.
    """
    data_path = Path(dataset_path)
    csv_files = list(data_path.glob("**/*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"Nie znaleziono plików CSV w katalogu: {dataset_path}")
        
    processed_datasets = {}
    
    for file_path in csv_files:
        house_name = file_path.stem  # np. 'House_12'

        # TODO: Remove after full mapping is done
        if house_name not in REFIT_CANONICAL_MAPPINGS:
            continue
        
        # Wczytanie pliku CSV
        df = pd.read_csv(file_path)
        
        # 1. Wydzielenie kolumny 'Time' jako index
        time_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
        if time_cols:
            time_col = time_cols[0]
            df[time_col] = pd.to_datetime(df[time_col])
            df.set_index(time_col, inplace=True)
        else:
            raise ValueError(f"Plik {file_path.name} nie zawiera kolumny czasu.")
            
        # 2. Usunięcie kolumny 'Unix'
        unix_cols = [col for col in df.columns if 'unix' in col.lower()]
        for u_col in unix_cols:
            if u_col in df.columns:
                df.drop(columns=[u_col], inplace=True)
                
        # Identyfikacja kolumn urządzeń
        appliance_cols = [col for col in df.columns if 'appliance' in col.lower() or 'Appliance' in col]
        
        # 3. Zastąpienie kolumny 'Aggregate' w zależności od flagi podanej parametrem
        if replace_aggregate_with_sum:
            if 'Aggregate' in df.columns:
                if appliance_cols:
                    # [KOMENTARZ / BŁĄD W PLANIE MERYTORYCZNYM]:
                    # Zastępowanie zmierzonego głównego agregatu (Aggregate) sumą podliczników (ApplianceX) 
                    # jest błędem w dziedzinie NILM. Licznik główny mierzy całkowity pobór domu (w tym 
                    # urządzenia niemonitorowane, oświetlenie, straty), więc suma IAM nigdy nie będzie równa Aggregate.
                    df['Aggregate'] = df[appliance_cols].sum(axis=1)
            else:
                raise ValueError(f"Brak kolumny 'Aggregate' w pliku {file_path.name}")
            
        # 4. Zamiana nazw kolumn ApplianceX na nazwy urządzeń według słownika
        mapping = REFIT_CANONICAL_MAPPINGS.get(house_name, REFIT_CANONICAL_MAPPINGS["default"])
        df.rename(columns=mapping, inplace=True)
        
        processed_datasets[house_name] = df
        
    return processed_datasets


# ==========================================
# KROK 6-7: Przerabianie na okna czasowe i dostęp do zbiorów
# ==========================================
def create_sliding_windows(df: pd.DataFrame, window_size: int = 100):
    """
    Przerabia zbiór danych na postać okien czasowych o podanym rozmiarze.
    
    [KOMENTARZ / BŁĄD W PLANIE TECHNICZNYM]:
    Plan opisuje: "w taki sposób aby okno 1 zawierało 1 wartość aggregate i wszystkie wartości dla urządzeń, 
    a dla okna 100 zawierało dodatkowo 99 starszych pomiarów aggregate."
    W uczeniu maszynowym (np. w sieciach 1D CNN / LSTM) wszystkie okna wejściowe muszą posiadać 
    STAŁY rozmiar (np. zawsze `window_size` kroków czasowych). Rosnące okno (od 1 do 99) powoduje zmienny 
    kształt (shape) danych wejściowych, co uniemożliwia formowanie batchów w PyTorch/TensorFlow. 
    Poniżej zaimplementowano standardowe okna o stałej długości (window_size).
    """
    agg_col = 'Aggregate'
    app_cols = [col for col in df.columns if col != agg_col]
    
    agg_values = df[agg_col].values
    apps_values = df[app_cols].values
    
    X, y = [], []
    
    # Generowanie okien czasowych o stałym rozmiarze window_size
    for i in range(window_size - 1, len(df)):
        # Okno wejściowe: historia agregatu o długości window_size
        window_agg = agg_values[i - window_size + 1 : i + 1]
        
        # Etykieta docelowa: pobór urządzeń w ostatnim punkcie okna czasowego
        target_apps = apps_values[i]
        
        X.append(window_agg)
        y.append(target_apps)
        
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


class NILMWindowDataset(Dataset):
    """
    Klasa zapewniająca dostęp do wszystkich podzielonych zbiorów danych według okien dla poszczególnych domostw.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        # Format wejściowy dla modeli w PyTorch: (Batch, Channels=1, Length)
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ==========================================
# GŁÓWNY SKRYPT URUCHOMIENIOWY
# ==========================================
if __name__ == "__main__":
    # Krok 1: Pobranie zbioru z Kaggle
    print("1. Pobieranie zbioru danych z Kaggle...")
    dataset_path = download_kaggle_dataset("data")
    print(f"Ścieżka do pobranych danych: {dataset_path}")
    
    # Krok 2-5: Iteracja, wczytanie, czyszczenie i mapowanie kolumn
    print("2-5. Wczytywanie i przetwarzanie plików CSV...")
    # Ustawiamy replace_aggregate_with_sum na False, aby zachować oryginalny pomiar z licznika głównego
    processed_houses = load_and_process_csv_files(dataset_path, replace_aggregate_with_sum=False)
    
    # Krok 6-7: Przerabianie na okna czasowe i udostępnienie zbiorów
    print("6-7. Generowanie okien czasowych i struktur Dataset...")
    window_size = 100
    house_datasets = {}
    
    for house_name, df_house in processed_houses.items():
        X, y = create_sliding_windows(df_house, window_size=window_size)
        
        # Udostępnienie przez klasę Dataset
        house_datasets[house_name] = NILMWindowDataset(X, y)
        print(f"  -> Domostwo [{house_name}]: Przygotowano {len(X)} okien czasowych.")

    # Przykład dostępu do przygotowanego zbioru dla konkretnego domu (np. House_12)
    if "House_12" in house_datasets:
        dataset_h12 = house_datasets["House_12"]
        sample_x, sample_y = dataset_h12[0]
        print(f"\n[Test dostępu] Kształt wejścia X (okno czasowe): {sample_x.shape}")
        print(f"[Test dostępu] Kształt wyjścia y (urządzenia): {sample_y.shape}")