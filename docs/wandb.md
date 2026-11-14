# Jak pracujemy z W&B

W&B (Weights & Biases) to strona, na której zapisują się wyniki naszych treningów.
Każdy trening wysyła tam swoje ustawienia i wykresy, a cały zespół widzi wszystko w jednym miejscu.

Ten dokument tłumaczy, jak z niego korzystać, żeby nasze wyniki dało się ze sobą porównać.

---

## Zanim zaczniesz (robisz to raz)

1. Poproś Adama o zaproszenie do zespołu **DL_NILM** na W&B i przyjmij je.
2. Na swoim komputerze, w folderze projektu, wpisz:
   ```bash
   uv sync
   uv run wandb login
   ```
   Program poprosi o klucz API. Znajdziesz go w ustawieniach swojego konta na stronie W&B.

Klucz jest jak hasło. Nie wklejaj go do kodu ani nie wysyłaj nikomu.

---

## Najważniejsze słowa

- **Run**: jedno uruchomienie treningu. Puściłeś trening, więc na W&B pojawia się jeden run.
- **Projekt**: miejsce, w którym leżą runy. Mamy dwa:
  - `nilm` na prawdziwe wyniki,
  - `nilm-dev` na próby i testy.
- **Config**: wszystkie ustawienia treningu (model, learning rate, rozmiar okna, domy...).
  Trzymamy go w pliku `.yaml` w folderze `configs/`.
- **Metryka**: liczba, którą śledzimy, np. błąd modelu.

---

## 7 zasad

### 1. Wszystkie wyniki trafiają do wspólnego zespołu

Nie musisz nic ustawiać, bo kod sam wysyła runy do zespołu **DL_NILM**.
Nie zmieniaj tego, nawet na chwilę. Run wysłany na prywatne konto jest niewidoczny dla reszty.

### 2. Trening uruchamiasz tylko przez `init_run`

W kodzie zawsze robisz tak:

```python
from nilm.config import load_config
from nilm.tracking import init_run

cfg = load_config("configs/moj_eksperyment.yaml")
run = init_run(cfg, job_type="train")
```

Nie pisz samodzielnie `wandb.init(...)`.

**Dlaczego?** `init_run` pilnuje, żeby każdy run miał komplet informacji: ustawienia, wersję kodu i miejsce zapisu.
Gdyby każdy pisał to po swojemu, każdemu czegoś by brakowało.

### 3. Każdy eksperyment ma swój plik w `configs/`

Chcesz coś przetestować? Skopiuj `configs/base.yaml`, nazwij go sensownie (np. `seq2point_kettle.yaml`) i zmień w nim to, co chcesz.

Nie zmieniaj parametrów bezpośrednio w kodzie.

**Dlaczego?** Wtedy na W&B widać dokładnie, z jakimi ustawieniami był puszczony każdy run.
Można też filtrować, np. „pokaż mi wszystkie runy z oknem 200”.

### 4. Zanim puścisz ważny trening, zrób commit

Zrób `git commit`, potem puść trening.

**Dlaczego?** W&B zapamiętuje, z której wersji kodu powstał wynik.
Jeśli masz niezapisane zmiany, ta informacja jest nieprawdziwa i za miesiąc nikt nie odtworzy wyniku.
Program wypisze wtedy ostrzeżenie:
`UWAGA: niezacommitowane zmiany`.

Do szybkich prób (zasada 7) commit nie jest potrzebny.

### 5. Metryki nazywamy zawsze tak samo

Wzór: **`zbiór/metryka/urządzenie`**

| Dobrze | Źle |
|---|---|
| `train/loss` | `loss_train` |
| `val/mae/kettle` | `kettle_mae`, `MAE kettle` |
| `test/f1/fridge_freezer` | `f1-fridge` |

- **zbiór**: `train`, `val` albo `test`
- **metryka**: małymi literami, np. `loss`, `mae`, `f1`
- **urządzenie**: nazwa jak w `mappings.py`, np. `kettle`, `fridge_freezer`

**Dlaczego?** Dla W&B `val/mae/kettle` i `kettle_mae` to dwie zupełnie różne rzeczy.
Jeśli każdy nazwie metrykę inaczej, nie da się położyć wyników dwóch osób obok siebie.

### 6. Wszyscy używamy tego samego podziału domów

Domy do treningu, walidacji i testu ustalamy wspólnie i nie zmieniamy ich po swojemu.

**Dlaczego?** Model testowany na domu, który „widział” podczas treningu, wypada dużo lepiej niż na nowym domu.
Jeśli każdy testuje na czymś innym, porównywanie wyników nie ma sensu.

### 7. Próby i testy nie trafiają do głównego projektu

Sprawdzasz tylko, czy kod działa? Masz dwie opcje:

- **Chcesz zobaczyć wykresy**: dopisz do swojego configu
  ```yaml
  wandb:
    debug: true
  ```
  Run trafi do `nilm-dev`, a nie do `nilm`.
- **Nie potrzebujesz W&B w ogóle**: uruchom z `WANDB_MODE=disabled`, np.
  ```bash
  WANDB_MODE=disabled uv run python scripts/train.py
  ```

**Dlaczego?** Główny projekt ma zawierać tylko wyniki, które coś znaczą.
Inaczej po tygodniu jest tam 300 runów, z czego 280 to „sprawdzam, czy się nie wywala”.

---

## Grupy i tagi, czyli jak zachować porządek

W configu możesz dopisać:

```yaml
name: s2p-kettle-seed1
wandb:
  group: s2p-kettle-w200
  tags: [seq2point, kettle]
```

- **`name`**: nazwa tego jednego runa.
- **`group`**: nazwa wspólna dla kilku runów tego samego eksperymentu.
  Przykład: puszczasz ten sam model 3 razy z różnym `seed`. Dajesz im tę samą grupę, a W&B pokaże średnią i rozrzut.
  Od razu widać, czy model jest naprawdę lepszy, czy po prostu miał szczęście.
- **`tags`**: etykiety, po których łatwo filtrować, np. `baseline`, `seq2point`.

### `job_type`, czyli co robi run

Przy `init_run` podajesz, czym jest run:

- `train`: trening modelu
- `eval`: ocena już wytrenowanego modelu
- `data`: przygotowanie danych
- `sweep`: automatyczne szukanie najlepszych parametrów

---

## Typowy dzień pracy

1. Kopiujesz `configs/base.yaml` do nowego pliku i zmieniasz ustawienia.
2. Ustawiasz `debug: true` i puszczasz krótki trening, żeby sprawdzić, czy działa.
3. Działa? Usuwasz `debug: true` i robisz commit.
4. Puszczasz prawdziwy trening. Wynik trafia do `nilm`.
5. Na stronie W&B porównujesz swój wynik z wynikami innych.
6. Masz ciekawe wnioski? Opisz je w **Report** na W&B. To dokument z wykresami, który łatwo pokazać zespołowi albo promotorowi.

---

## Gdy coś nie działa

| Problem | Co zrobić |
|---|---|
| `wandb: ERROR ... not logged in` | `uv run wandb login` |
| `permission denied` / brak dostępu do `DL_NILM` | nie jesteś jeszcze w zespole, poproś o zaproszenie |
| Komputer bez internetu (np. klaster) | uruchom z `WANDB_MODE=offline`, a potem na komputerze z internetem `uv run wandb sync wandb/offline-run-*` |
| `Nieznany klucz konfiguracji` | literówka w pliku `.yaml`; sprawdź nazwę w `src/nilm/config.py` |
