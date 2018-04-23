## Pisanie funkcji `show_correctness`

Funkcje piszemy w ogólnym schemacie tak:
```python
def nazwa_funkcji(argument1, argument2):
    # coś robimy z argumentami
    # coś zwracamy np.:
    return argument1
```

przykład funkcji, która dodaje:
```python
def dodaj(x, y):
    suma = x + y
    return suma
```

w notebooku jest prośba o funkcję która przyjmuje argumenty `model`, `X_test`
oraz `y_test` i nazywa się `show_correctness`. W związku z tym zaczniemy ją pisać tak:
```python
def show_correctness(model, X_test, y_test):
    # tutaj coś więcej
```

co należy wpisać do środka funkcji?
Są to trzy linijki, które wykorzystywaliśmy wcześniej w notebooku do:
* generowania predykcji
* liczenia poprawności
* wyświetlania poprawności

musicie te linijki skopiować do funkcji z drobnymi zmianami:
* linijki muszą mieć odpowiednie wcięcie (np. wszytkie jeden tab) tak aby python wiedział, że te linijki są częścią funkcji.
* zamiast np. `logistic_model` musicie korzystać z nazw argumentów podanych funkcji tzn w tym wypadku `model`
* tak samo w przypadku pozostałych dwóch argumentów funkcji: `X_test` oraz `y_test`
