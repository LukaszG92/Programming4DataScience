"""
In this exercise, you will create a Python class called `FactorialGenerator` that implements the iterator-iterable pattern to generate factorial numbers.
The `FactorialGenerator` class should allow users to iterate through a sequence of factorial numbers starting from 1! (1 factorial) and increasing by 1 each time.
The `FactorialGenerator` class should have the following features:
    1. Initialize the generator with the maximum number of factorial numbers to generate (`n`).
    2. Implement the `__iter__` method to return an iterator object (which can be the `FactorialGenerator` object itself).
    3. Implement the `__next__` method in the iterator class (you can call it `FactorialIterator`). This method should calculate and return the next factorial number
       in the sequence. The sequence should stop after generating `n` factorial numbers.
    4. Implement a method called `reset` that allows users to reset the generator to its initial state.
"""

# Punto 1 e Punto 2
class FactorialGenerator:
    def __init__(self, n):
        self.n = n
        self.iterator = None

    def __iter__(self):
        self.iterator = FactorialIterator(self.n)
        return self.iterator

    def reset(self):
        if self.iterator:
            self.iterator.reset()


# Punto 3
class FactorialIterator:
    def __init__(self, n):
        self.n = n
        self.current = 0
        self.factorial = 1

    def __iter__(self):
        return self

    def __next__(self):
        if self.current >= self.n:
            raise StopIteration

        if self.current == 0:
            self.current += 1
            return 1

        self.factorial *= self.current
        self.current += 1
        return self.factorial

    def reset(self):
        self.current = 0
        self.factorial = 1

# Punto 4
fact_gen = FactorialGenerator(5)

print("Prima iterazione:", end=" ")
for factorial in fact_gen:
    print(factorial, end=" ")

print("\n\nReset e seconda iterazione:", end=" ")
fact_gen.reset()
for factorial in fact_gen:
    print(factorial, end=" ")
