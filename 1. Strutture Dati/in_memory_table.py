"""
In this exercise, you will create a Python program that simulates an in-memory data table for storing information about students. The data table will be represented
using lists and dictionaries. Your program should allow users to perform the following operations:
    1. Add a new student to the data table with the following information:
        - Student ID (a unique identifier)
        - First name
        - Last name
        - Age
        - GPA
    2. Update the information of an existing student by providing their Student ID.
    3. Delete a student from the data table by providing their Student ID.
    4. List all students in the data table, displaying their information in a tabular format.
"""

from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from tabulate import tabulate


@dataclass
class Student:
    student_id: str
    first_name: str
    last_name: str
    age: int
    gpa: float


class StudentDatabase:
    def __init__(self):
        self.students: Dict[str, Student] = {}

    def add_student(self,
                    student_id: str,
                    first_name: str,
                    last_name: str,
                    age: int,
                    gpa: float) -> bool:

        # Controlla se l'ID dello studente è già registrato
        if student_id in self.students:
            print(f"Error: Student ID {student_id} already exists!")
            return False

        # Crea un nuovo studente
        student = Student(
            student_id=student_id,
            first_name=first_name,
            last_name=last_name,
            age=age,
            gpa=gpa
        )

        # Aggiunta al database
        self.students[student_id] = student
        return True

    def update_student(self,
                       student_id: str,
                       **kwargs) -> bool:

        # Controlla se l'utente esiste
        if student_id not in self.students:
            print(f"Error: Student ID {student_id} not found!")
            return False

        student = self.students[student_id]

        # Aggiorna i campi modificati
        for field, value in kwargs.items():
            if field not in ['first_name', 'last_name', 'age', 'gpa']:
                print(f"Warning: Invalid field {field} ignored")
                continue

            setattr(student, field, value)

        return True

    def delete_student(self, student_id: str) -> bool:

        if student_id not in self.students:
            print(f"Error: Student ID {student_id} not found!")
            return False

        del self.students[student_id]
        return True

    def list_students(self) -> None:

        if not self.students:
            print("No students in database.")
            return

        # Converte gli oggetti Studente in una lista
        headers = ["ID", "First Name", "Last Name", "Age", "GPA"]
        rows = [[s.student_id, s.first_name, s.last_name, s.age, f"{s.gpa:.2f}"]
                for s in self.students.values()]

        # Ordina gli studenti per ID
        rows.sort(key=lambda x: x[0])

        # Stampa la tabella
        print(tabulate(rows, headers=headers, tablefmt="grid"))

    def get_student(self, student_id: str) -> Optional[Student]:
        return self.students.get(student_id)


db = StudentDatabase()

# Aggiungi un po' di studenti al Database
db.add_student("001", "John", "Doe", 20, 3.5)
db.add_student("002", "Jane", "Smith", 22, 3.8)
db.add_student("003", "Bob", "Johnson", 19, 3.2)

print("Initial database:")
db.list_students()

# Aggiorna uno studente
print("\nUpdating student 002's GPA...")
db.update_student("002", gpa=4.0)
db.list_students()

# Rimuove uno studente
print("\nDeleting student 003...")
db.delete_student("003")
db.list_students()
