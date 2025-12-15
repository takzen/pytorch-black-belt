# 🔥 PyTorch Advanced Course - Deep Dive Roadmap

Kompleksowy kurs zaawansowanego PyTorch dla inżynierów ML, którzy chcą zrozumieć, jak naprawdę działa deep learning pod maską.

---

## 🗝️ Moduł 1: Głębokie zanurzenie w Tensory (Fundamenty)

**Jak PyTorch zarządza pamięcią i matematyką.**

- **01_Storage_vs_View.ipynb** – Storage vs Tensor. Zrozumienie, czym jest Stride (krok) i dlaczego `view()` rzuca błędy na nieciągłych tensorach (contiguous).
- **02_Broadcasting_Magic.ipynb** – Matematyka rozgłaszania. Jak dodawać wektor do macierzy bez pętli for i kopiowania pamięci.
- **03_Einsum_Is_All_You_Need.ipynb** – `torch.einsum`. Jedna funkcja, by rządzić wszystkimi (mnożenie macierzy, iloczyny skalarne, transpozycje w jednym stringu).
- **04_Advanced_Indexing.ipynb** – `gather`, `scatter` i `index_select`. Jak manipulować danymi w Reinforcement Learning i Grafach (gdzie indeksy są skomplikowane).
- **05_In_Place_Operations.ipynb** – `x += 1` vs `x = x + 1`. Kiedy oszczędzasz pamięć, a kiedy niszczysz historię gradientów?
- **06_Einops_Tutorial.ipynb** – Biblioteka einops. Nowoczesne, czytelne manipulacje tensorami (`rearrange`, `reduce`), które zastępują skomplikowane `view`/`permute`.
- **07_Named_Tensors.ipynb** – Eksperymentalna funkcja: Tensory z nazwami wymiarów (np. `img.rename(C, H, W)`). Bezpieczeństwo typów w Deep Learningu.

---

## 🧮 Moduł 2: Wnętrze Autogradu (Silnik)

**Jak działa różniczkowanie automatyczne i jak je hackować.**

- **08_Computational_Graph_Viz.ipynb** – Wizualizacja DAG (Directed Acyclic Graph). Czym są liście (`is_leaf`) i funkcje `grad_fn`.
- **09_Requires_Grad_Mechanics.ipynb** – Kiedy używać `.detach()`, `with torch.no_grad()` a kiedy `inference_mode()`. Subtelne różnice w wydajności.
- **10_Custom_Autograd_Function.ipynb** – Pisanie własnej warstwy z ręczną metodą `backward()`. (Np. dla funkcji, której PyTorch nie obsługuje lub dla optymalizacji).
- **11_Jacobian_and_Hessian.ipynb** – Obliczanie pochodnych wyższego rzędu (np. do meta-learningu MAML) za pomocą `torch.autograd.functional`.
- **12_Retain_Graph_Trick.ipynb** – Błąd "Trying to backward through the graph a second time". Kiedy i dlaczego musimy używać `retain_graph=True`?
- **13_Gradient_Accumulation.ipynb** – Jak trenować na Batch Size = 128, mając pamięć tylko na 8? Symulacja dużych batchy.
- **14_Forward_Mode_AD.ipynb** – Nowość w AI. Różniczkowanie w przód (Forward Mode) vs klasyczne wstecz (Reverse Mode). Kiedy to się przydaje?

---

## 💿 Moduł 3: Inżynieria Danych (Paliwo)

**Optymalizacja pipeline'u danych, żeby GPU nie czekało.**

- **15_Dataset_vs_IterableDataset.ipynb** – Kiedy dane nie mieszczą się w RAM. Streaming danych z dysku/sieci.
- **16_Custom_Collate_Fn.ipynb** – Obsługa danych o różnej długości (np. tekst, audio). Jak pisać własne funkcje sklejające batch.
- **17_Samplers_and_Imbalance.ipynb** – `WeightedRandomSampler`. Jak trenować na niezbalansowanych danych bez duplikowania plików.
- **18_Num_Workers_and_Pin_Memory.ipynb** – Analiza wielowątkowości w DataLoaderze. Czym jest Page-Locked Memory (`pin_memory`) i kiedy przyspiesza transfer na GPU.
- **19_Data_Augmentation_GPU.ipynb** – Kornia vs Torchvision. Dlaczego augmentacja na CPU (w DataLoaderze) to wąskie gardło i jak przenieść ją na GPU.
- **20_WebDataset_Concept.ipynb** – (Teoria/Demo) Format TAR do ultra-szybkiego czytania milionów małych plików (standard w treningu LLM/Stable Diffusion).

---

## 🧠 Moduł 4: Zaawansowana Architektura (Konstrukcja)

**Triki architektoniczne i zarządzanie stanem modelu.**

- **21_Module_Life_Cycle.ipynb** – `__init__`, `forward`, `__call__`. Jak działa magia `nn.Module` pod spodem.
- **22_Buffers_vs_Parameters.ipynb** – Czym się różni `self.param` od `register_buffer`? (Przykład na BatchNorm i Positional Encoding).
- **23_Hooks_Anatomy.ipynb** – `register_forward_hook` i `register_backward_hook`. Jak wyciągać aktywacje z środka sieci bez zmieniania jej kodu (Feature Extraction).
- **24_Weight_Initialization.ipynb** – Dlaczego `kaiming_normal` i `xavier_uniform` są kluczowe? Wizualizacja eksplozji/zaniku gradientu przy złej inicjalizacji.
- **25_Weight_Sharing.ipynb** – Jak użyć tej samej warstwy w dwóch miejscach sieci (np. w Autoenkoderach Tied-Weights).
- **26_Dynamic_Control_Flow.ipynb** – Używanie pętli `for` i `if` wewnątrz `forward`. Jak PyTorch radzi sobie z dynamicznymi grafami (w przeciwieństwie do TensorFlow).
- **27_Gradient_Checkpointing.ipynb** – Handel: Czas za Pamięć. Jak zmieścić 10x większy model w VRAM, obliczając część grafu dwukrotnie.
- **28_Model_Surgery.ipynb** – Wczytywanie pretrenowanego modelu i podmienianie jego warstw (np. zmiana rozmiaru wejścia w ResNet).

---

## ⚡ Moduł 5: Trening i Optymalizacja (Szybkość)

**Stabilizacja i przyspieszanie uczenia.**

- **29_Optimizer_Internals.ipynb** – Jak działa `torch.optim`? Pisanie własnego optymalizatora od zera (SGD z Momentum).
- **30_Learning_Rate_Schedulers.ipynb** – `CosineAnnealing`, `OneCycleLR`, `ReduceLROnPlateau`. Wizualizacja wpływu na zbieżność.
- **31_Mixed_Precision_AMP.ipynb** – `torch.cuda.amp`. Jak używać Autocast i GradScaler, żeby trenować 2x szybciej w FP16.
- **32_Gradient_Clipping.ipynb** – Jak zapobiegać NaN w treningu (szczególnie w RNN/Transformerach) poprzez przycinanie normy gradientu.
- **33_Torch_Compile_Intro.ipynb** – PyTorch 2.0. Wprowadzenie do `torch.compile()` i trybów optymalizacji (`reduce-overhead`, `max-autotune`).
- **34_Bottleneck_Analysis.ipynb** – Jak używać `torch.autograd.profiler`, żeby sprawdzić, która warstwa zjada najwięcej czasu.
- **35_Weight_Decay_vs_L2.ipynb** – Subtelna różnica między Weight Decay w AdamW a regularyzacją L2 (i dlaczego AdamW jest lepszy).
- **36_Reproducibility_Seeding.ipynb** – Jak poprawnie ustawić ziarna losowości (`manual_seed`, `deterministic`), żeby wynik był zawsze ten sam (również na GPU).

---

## 📦 Moduł 6: Ekosystem i Produkcja (Skala)

**Narzędzia dojrzałego inżyniera.**

- **37_PyTorch_Lightning_Refactor.ipynb** – Przepisanie pętli treningowej na `LightningModule`. Czysty kod bez boilerplate'u.
- **38_TensorBoard_Logging.ipynb** – Jak logować nie tylko stratę, ale też histogramy wag, obrazy i graf modelu do TensorBoarda.
- **39_TorchScript_Tracing.ipynb** – `torch.jit.trace`. Zamiana dynamicznego modelu w statyczny graf dla C++. Ograniczenia i pułapki.
- **40_TorchScript_Scripting.ipynb** – `torch.jit.script`. Jak kompilować modele z logiką sterowania (`if`/`else`), której Tracing nie widzi.
- **41_ONNX_Advanced_Export.ipynb** – Dynamiczne osie w ONNX (zmienna długość batcha). Debugowanie błędów eksportu.
- **42_Inference_Optimization.ipynb** – Łączenie warstw (Conv+BN fusion) przed wdrożeniem dla szybszego działania.
- **43_DDP_Concepts.ipynb** – Teoria treningu rozproszonego (Distributed Data Parallel). Jak działa synchronizacja gradientów między wieloma GPU.
- **44_FSDP_Concepts.ipynb** – Fully Sharded Data Parallel. Jak trenować modele, które nie mieszczą się na jednej karcie (dzielenie modelu na kawałki).

---

## 🎮 Moduł 7: Eksperymenty i Ciekawostki (Bonus)

**Rzeczy dziwne i przydatne.**

- **45_Meta_Learning_Higher.ipynb** – Użycie biblioteki `higher` do różniczkowania przez pętlę optymalizatora (Unrolled optimization).
- **46_PyTorch_Hooks_Visualization.ipynb** – Wykorzystanie hooków do wizualizacji map aktywacji (CAM - Class Activation Mapping).
- **47_Adversarial_Example_Generation.ipynb** – Użycie dostępu do gradientów wejścia, aby stworzyć obraz mylący sieć (FGSM).
- **48_Neural_Style_Transfer_Raw.ipynb** – Manipulacja aktywacjami wewnątrz VGG do przenoszenia stylu artystycznego (bez gotowych bibliotek).
- **49_Custom_Loss_Functions.ipynb** – Pisanie złożonych funkcji kosztu (np. Triplet Loss, Contrastive Loss) z wykorzystaniem operacji macierzowych.
- **50_The_Grand_Exam.ipynb** – "Egzamin Końcowy". Zestaw trudnych pytań rekrutacyjnych i snippetów kodu do debugowania ("Znajdź błąd w tej pętli treningowej").

---

**Powodzenia w zgłębianiu PyTorch! 🔥**
