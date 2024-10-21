"""
In this exercise, you will create a Python class that represents a simple iterable data structure. You will implement both the
`__iter__` and `__next__` methods to make the class iterable.
    1. Create a class called `NumberSeries` that initializes with two integers, `start` and `end`. The class should represent a series
       of numbers starting from `start` and ending at `end` (inclusive).
    2. Implement the `__iter__` method in the `NumberSeries` class. This method should return an iterator object (which can be the
       `NumberSeries` object itself).
    3. Implement the `__next__` method in the iterator class (you can call it `NumberSeriesIterator`). This method should return the next
    number in the series. If the current number exceeds the `end` value, it should raise `StopIteration` to signal the end of iteration.
    4. Create an instance of the `NumberSeries` class with a specific range (e.g., `NumberSeries(1, 5)`). 5. Use a `for` loop to iterate
    through the numbers in the series and print each number.
"""

# Punto 1
class NumberSeriesSelf:
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.current = None

    def __iter__(self):
        self.current = self.start
        return self

    def __next__(self):
        if self.current > self.end:
            raise StopIteration
        result = self.current
        self.current += 1
        return result


#Punto 2
class NumberSeries:
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __iter__(self):
        return NumberSeriesIterator(self.start, self.end)


# Punto 3
class NumberSeriesIterator:
    def __init__(self, start, end):
        self.current = start
        self.end = end

    def __iter__(self):
        return self

    def __next__(self):
        if self.current > self.end:
            raise StopIteration
        result = self.current
        self.current += 1
        return result


# Punto 4
print("Usando la classe del Punto 1:" , end=' ')
# Punto 5
series2 = NumberSeriesSelf(1, 5)
for num in series2:
    print(num, end=' ')

# Punto 4
print("\n\nUsando le classi dei Punti 2 e 3:", end=' ')
# Punto 5
series1 = NumberSeries(1, 5)
for num in series1:
    print(num, end=' ')


