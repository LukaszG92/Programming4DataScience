from functools import reduce
from typing import List, Dict, Union


def extract_sales_amounts(sales_data: List[Dict[str, Union[str, float, int]]]) -> List[float]:
    return list(map(lambda sale: float(sale["amount"]), sales_data))


def calculate_total_sales(amounts: List[float]) -> float:
    return round(reduce(lambda x, y: x + y, amounts, 0), 2)


def process_sales_data(sales_data: List[Dict[str, Union[str, float, int]]]) -> float:
    amounts = extract_sales_amounts(sales_data)
    return calculate_total_sales(amounts)


if __name__ == "__main__":
    # Example usage with sample data
    sample_sales_data = [
        {"id": 1, "amount": 150.75, "product": "Laptop", "date": "2024-01-01"},
        {"id": 2, "amount": 899.99, "product": "Smartphone", "date": "2024-01-02"},
        {"id": 3, "amount": 49.99, "product": "Headphones", "date": "2024-01-02"},
        {"id": 4, "amount": 299.50, "product": "Monitor", "date": "2024-01-03"}
    ]

    # Process the data
    total_sales = process_sales_data(sample_sales_data)

    print("\nExample usage results:")
    print(f"Sales amounts: {extract_sales_amounts(sample_sales_data)}")
    print(f"Total sales: ${total_sales:,.2f}")