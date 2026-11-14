# Jak wygląda dzień pracy

Ten dokument pokazuje krok po kroku, jak pracujemy nad projektem: od włączenia komputera do oddania wyników zespołowi.

Nie musisz znać wszystkiego na pamięć. Na końcu jest **ściąga**, wystarczy do niej zaglądać.

Jeśli pracujesz z nami pierwszy raz, najpierw zrób kroki z sekcji „Zanim zaczniesz” w [wandb.md](wandb.md).

---

## Krok 1. Zaktualizuj projekt

Zanim cokolwiek zmienisz, pobierz to, co zrobili inni:

```bash
git checkout main
git pull
uv sync
```

- `git pull` pobiera najnowszy kod.
- `uv sync` doinstalowuje biblioteki, jeśli ktoś dodał nowe.

**Dlaczego?** Jeśli zaczniesz pracę na starym kodzie, później trudno będzie połączyć Twoje zmiany ze zmianami innych.

---

## Krok 2. Załóż własną gałąź

Każde zadanie robisz na osobnej gałęzi (branchu), nigdy bezpośrednio na `main`.

```bash
git checkout -b seq2point-kettle
```

Nazwa gałęzi mówi krótko, nad czym pracujesz, np. `seq2point-kettle`, `fix-resampling`, `eda-gaps`.

**Dlaczego?** `main` to wersja, na której wszyscy polegają. Na swojej gałęzi możesz psuć do woli, nikomu nie przeszkadzając.

---

## Krok 3. Przygotuj eksperyment

Skopiuj bazowy config i nazwij go tak, żeby było wiadomo, co sprawdzasz:

```bash
cp configs/base.yaml configs/seq2point_kettle.yaml
```

W nowym pliku zmień to, co chcesz przetestować, np. model, rozmiar okna czy learning rate.

Ustawienia zmieniasz **w pliku `.yaml`**, a nie w kodzie.

**Dlaczego?** Cały config trafia do W&B. Za miesiąc dokładnie zobaczysz, z jakimi ustawieniami powstał wynik.

---

## Krok 4. Sprawdź, czy w ogóle działa

Zanim puścisz długi trening, sprawdź na szybko, czy kod się nie wywala. Dopisz do configu:

```yaml
wandb:
  debug: true
```

Ustaw też mało epok, np. `epochs: 1`, i uruchom trening:

```bash
uv run python scripts/train.py configs/seq2point_kettle.yaml
```

> Skrypt `scripts/train.py` jest jeszcze w przygotowaniu. Gdy powstanie, będzie się uruchamiał właśnie tak.

Wynik trafi do projektu **`nilm-dev`** na W&B, czyli do „brudnopisu”.

Poprawiaj i uruchamiaj tyle razy, ile trzeba. Na tym etapie nie musisz niczego commitować.

**Dlaczego?** Główny projekt `nilm` ma zawierać tylko prawdziwe wyniki. Inaczej utoną w setkach prób.

---

## Krok 5. Zapisz zmiany w gicie

Działa? To teraz:

1. Usuń z configu `debug: true`.
2. Przywróć docelową liczbę epok.
3. Zrób commit:

```bash
git add .
git commit -m "Seq2point dla czajnika, okno 200"
```

**Dlaczego?** W&B zapisuje, z której wersji kodu powstał wynik. Bez commita ta informacja jest nieprawdziwa i wyniku nie da się odtworzyć.

Jeśli zapomnisz, program wypisze ostrzeżenie `UWAGA: niezacommitowane zmiany`.

---

## Krok 6. Puść prawdziwy trening

```bash
uv run python scripts/train.py configs/seq2point_kettle.yaml
```

Teraz wynik trafia do głównego projektu **`nilm`**.

**Dobra praktyka:** puść ten sam eksperyment 2–3 razy z różnym `seed` i daj im wspólną grupę:

```yaml
name: s2p-kettle-seed1      # przy kolejnych: seed2, seed3
train:
  seed: 1                   # przy kolejnych: 2, 3
wandb:
  group: s2p-kettle-w200    # taka sama dla wszystkich trzech
  tags: [seq2point, kettle]
```

**Dlaczego?** Jeden run może mieć po prostu szczęście. Gdy masz trzy, W&B pokaże średnią i rozrzut, i widać, czy poprawa jest prawdziwa.

---

## Krok 7. Obejrzyj wyniki

Na stronie W&B, w projekcie `DL_NILM/nilm`:

- znajdź swoje runy, np. filtrując po tagu albo grupie,
- porównaj je z wynikami innych na tych samych metrykach, np. `test/mae/kettle`,
- sprawdź wykresy uczenia: czy `train/loss` spada? Czy `val/...` nie zaczyna rosnąć (przeuczenie)?

Pamiętaj: porównujemy tylko wyniki na **tym samym podziale domów**. Inaczej porównanie nic nie znaczy.

---

## Krok 8. Podziel się wynikami

### Kod: pull request

```bash
git push -u origin seq2point-kettle
```

Potem na GitHubie:

1. Otwórz **pull request** do `main`.
2. W opisie napisz krótko: co zmieniłeś i jaki był wynik (z linkiem do runa w W&B).
3. Poproś kogoś z zespołu o przejrzenie.
4. Po akceptacji zrób merge.

**Dlaczego?** Druga para oczu wyłapuje błędy, a reszta zespołu wie, co się zmieniło.

### Wnioski: raport w W&B

Masz ciekawy wynik albo coś zaskakującego? Opisz to w **Report** na W&B. To dokument z wykresami prosto z runów, który łatwo pokazać zespołowi albo promotorowi.

---

## Czego unikać

| Nie rób tak | Zamiast tego |
|---|---|
| Praca bezpośrednio na `main` | własna gałąź (krok 2) |
| Zmiana parametrów w kodzie | zmiana w pliku `.yaml` (krok 3) |
| Próby i testy w projekcie `nilm` | `debug: true`, czyli projekt `nilm-dev` (krok 4) |
| Prawdziwy trening bez commita | najpierw commit, potem trening (krok 5) |
| Wniosek z jednego runa | 2–3 seedy w jednej grupie (krok 6) |
| Własny podział domów | wspólny podział ustalony w zespole |
| Wklejanie klucza W&B do kodu | `uv run wandb login`, klucz zostaje tylko u Ciebie |

---

## Ściąga

```bash
# 1. Zaktualizuj
git checkout main && git pull && uv sync

# 2. Nowa gałąź
git checkout -b <nazwa-zadania>

# 3. Nowy eksperyment
cp configs/base.yaml configs/<eksperyment>.yaml

# 4. Próba (w configu: wandb.debug: true, mało epok)
uv run python scripts/train.py configs/<eksperyment>.yaml

# 5. Commit (po usunięciu debug: true)
git add . && git commit -m "<co zrobiłeś>"

# 6. Prawdziwy trening
uv run python scripts/train.py configs/<eksperyment>.yaml

# 8. Oddaj pracę
git push -u origin <nazwa-zadania>      # potem pull request na GitHubie
```

Coś nie działa? Zajrzyj do sekcji „Gdy coś nie działa” w [wandb.md](wandb.md).
