# 🥋 PyTorch Black Belt

> **Od `import torch` do Mistrzostwa Inżynierskiego.**
> Kompleksowa kolekcja notebooków zaprojektowanych, aby wypełnić lukę między "uruchomieniem modelu" a zrozumieniem mechanizmu pod maską.

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white) ![Status](https://img.shields.io/badge/Status-Aktywny_Rozwój-success?style=for-the-badge)

## 🎯 Cel

Większość tutoriali kończy się na `model.fit()`. To repozytorium zaczyna tam, gdzie one kończą.
**PyTorch Black Belt** to zrozumienie "dlaczego" i "jak":

- Dlaczego `.view()` rzuca błędy na nieciągłych tensorach?
- Jak napisać własne przejście wsteczne (Backward) dla nowej operacji?
- Jak debugować ciche awarie, które nie rzucają wyjątków, ale rujnują trening?
- Jak zoptymalizować potoki danych, gdy wykorzystanie GPU jest niskie?

## 📚 Program nauczania: Ścieżka do Mistrzostwa

### 🗝️ Moduł 1: Głębokie zanurzenie w Tensory (Fundamenty)

_Nie opanujesz modelu, jeśli nie opanujesz struktury danych._

- **Dojo Broadcastingu:** Matematyka stojąca za rozszerzaniem wymiarów.
- **Kroki i Pamięć:** Zrozumienie `storage()`, `view()` vs `reshape()` oraz `contiguous()`.
- **Einsum:** Magiczna funkcja zastępująca złożone operacje macierzowe.
- **Operacje w miejscu:** Kiedy `x += 1` oszczędza pamięć, a kiedy psuje Autograd.

### 🧮 Moduł 2: Wnętrze Autogradu (Silnik)

_Hakowanie silnika pochodnych._

- **Graf Obliczeniowy:** Wizualizacja dynamicznej konstrukcji grafu.
- **`retain_graph=True`:** Przypadki użycia wykraczające poza podstawy.
- **Niestandardowe Funkcje Autogradu:** Pisanie własnych metod `forward` i `backward`.
- **Akumulacja Gradientów:** Symulowanie dużych batch'y na małym VRAM.

### 💿 Moduł 3: Inżynieria Danych (Paliwo)

_Śmieci na wejściu, śmieci na wyjściu. Wolne wejście, wolny trening._

- **IterableDataset:** Obsługa strumieni i zbiorów danych większych niż RAM.
- **Niestandardowy Collate:** Zarządzanie sekwencjami o zmiennej długości i padding w locie.
- **Zaawansowane Próbkowanie:** Dynamiczne balansowanie niezbalansowanych zbiorów danych.
- **Analiza Wąskich Gardeł:** Optymalizacja `num_workers`, `pin_memory` i prefetchingu.

### 🧠 Moduł 4: Zaawansowana Architektura (Konstrukcja)

_Budowanie solidnych i złożonych systemów._

- **Hooki:** Inspekcja aktywacji i gradientów wewnątrz czarnej skrzynki.
- **Strategie Inicjalizacji:** Dlaczego Xavier i Kaiming mają znaczenie dla zbieżności.
- **Współdzielenie Wag:** Wiązanie parametrów między warstwami (np. Autokodery).
- **Dynamiczny Przepływ Sterowania:** Używanie logiki Pythona (`if/else`) wewnątrz grafu.

### ⚡ Moduł 5: Trening i Optymalizacja (Szybkość)

_Wyciskanie każdego FLOPS-a z twojego GPU._

- **Mieszana Precyzja (AMP):** Implementacja `fp16` dla 2x przyspieszenia.
- **Schedulery:** Rozgrzewka, Cosine Annealing i Cykliczne Tempo Uczenia.
- **Przycinanie Gradientów:** Zapobieganie eksplodującym gradientom w RNN/Transformerach.
- **Torch 2.0:** Opanowanie `torch.compile` i strategii fuzji.

### 📦 Moduł 6: Ekosystem i Produkcja (Skala)

_Przejście z notebooka do klastra._

- **PyTorch Lightning:** Strukturyzowanie kodu dla powtarzalności.
- **TorchScript i Tracing:** Eksportowanie modeli do środowisk C++.
- **DDP (Distributed Data Parallel):** Mechanika treningu na wielu GPU.
- **Profilowanie:** Używanie PyTorch Profiler do znajdowania wąskich gardeł w kodzie.

## 🛠️ Stos Technologiczny i Narzędzia

Ten projekt koncentruje się na nowoczesnym, wydajnym ekosystemie PyTorch:

- **Python 3.10+**
- **PyTorch 2.x** (Główny framework, koncentracja na `torch.compile` i dynamicznych grafach)
- **Einops** (Czytelne i potężne operacje tensorowe)
- **PyTorch Lightning** (Organizacja złożonych potoków treningowych)
- **Torch Profiler i TensorBoard** (Debugowanie wydajności)
- **NumPy i Pandas** (Manipulacja danymi)
- **Matplotlib i Seaborn** (Wizualizacja wnętrz i krajobrazów strat)

## 🚀 Jak Używać

Masz dwie opcje: natychmiastowe wykonanie w chmurze lub profesjonalną konfigurację lokalną.

### ☁️ Opcja 1: Google Colab (Zero Konfiguracji)

Najszybszy sposób na naukę. Każdy notebook w tym repozytorium ma przycisk **"Open in Colab"** na górze.

1.  Otwórz dowolny plik `.ipynb` z listy plików.
2.  Kliknij przycisk <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" style="vertical-align: middle">.
3.  Kod uruchamia się natychmiast na darmowych GPU Google.

### 💻 Opcja 2: Rozwój Lokalny (VS Code + uv)

Zalecane dla inżynierów budujących własne środowisko eksperymentalne.

1.  **Sklonuj repozytorium:**

    ```bash
    git clone https://github.com/takzen/pytorch-black-belt.git
    cd pytorch-black-belt
    ```

2.  **Zainicjalizuj środowisko z `uv`:**

    ```bash
    # Utwórz wirtualne środowisko
    uv venv

    # Aktywuj je:
    # Windows:
    .venv\Scripts\activate
    # Linux/Mac:
    source .venv/bin/activate
    ```

3.  **Zainstaluj Zależności (Stos Inżynierski):**

    ```bash
    # 1. Zainstaluj PyTorch ze wsparciem CUDA (Dostosuj index-url dla twojego GPU)
    uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

    # 2. Zainstaluj Narzędzia (Einops, Profilery, Wizualizacja)
    uv pip install numpy pandas matplotlib seaborn einops lightning tensorboard torch-tb-profiler jupyterlab ipywidgets
    ```

---

## 📊 Statystyki Projektu

- **Kompleksowy program nauczania** skupiający się wyłącznie na wewnętrznych mechanizmach PyTorch i inżynierii.
- **Od Matematyki do Produkcji:** Od ręcznej implementacji propagacji wstecznej do treningu rozproszonego (DDP).
- **6 Modułów Głębokiego Zanurzenia:** Tensory, Autograd, Inżynieria Danych, Architektura, Optymalizacja, Produkcja.
- **Implementacje Referencyjne:** Niestandardowe kernele CUDA, Memory-efficient Attention, Gradient Checkpointing.
- **Nowoczesny PyTorch 2.0:** Wykorzystanie `torch.compile` i strategii fuzji.

---

**Autor:** Krzysztof Pika
