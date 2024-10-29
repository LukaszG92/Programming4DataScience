import numpy as np
from typing import List, Tuple
import os


def create_sample_data():
    q1_data = """
        2000,2500,3000
        1500,1800,2200
        3500,3800,4000
        2800,3000,3200"""

    q2_data = """
        3200,3500,3800
        2400,2600,2800
        4200,4500,4800
        3400,3600,3800"""

    q3_data = """
        4000,4200,4500
        3000,3200,3400
        5000,5200,5400
        4000,4200,4400"""

    with open('sales_data_q1.txt', 'w') as f:
        f.write(q1_data)
    with open('sales_data_q2.txt', 'w') as f:
        f.write(q2_data)
    with open('sales_data_q3.txt', 'w') as f:
        f.write(q3_data)


def load_quarterly_data(filename: str) -> np.ndarray:
    return np.loadtxt(filename, delimiter=',')


def merge_quarterly_data(filenames: List[str]) -> np.ndarray:
    quarterly_data = [load_quarterly_data(f) for f in filenames]

    return np.hstack(quarterly_data)


def calculate_statistics(sales_data: np.ndarray) -> Tuple[np.ndarray, ...]:
    transposed_data = sales_data.T

    total_sales_per_product = np.sum(sales_data, axis=1)
    avg_monthly_sales_per_product = np.mean(sales_data, axis=1)

    total_sales_per_month = np.sum(sales_data, axis=0)
    avg_sales_per_month = np.mean(sales_data, axis=0)

    return (
        transposed_data,
        total_sales_per_product,
        avg_monthly_sales_per_product,
        total_sales_per_month,
        avg_sales_per_month
    )


def print_results(
        sales_data: np.ndarray,
        transposed_data: np.ndarray,
        total_sales_per_product: np.ndarray,
        avg_monthly_sales_per_product: np.ndarray,
        total_sales_per_month: np.ndarray,
        avg_sales_per_month: np.ndarray
) -> None:
    num_products = sales_data.shape[0]
    num_months = sales_data.shape[1]

    print("\n=== Sales Data Analysis Results ===")

    print("\n1. Original Sales Matrix (Products × Months):")
    print(sales_data)

    print("\n2. Transposed Sales Matrix (Months × Products):")
    print(transposed_data)

    print("\n3. Total Sales by Product:")
    for i, total in enumerate(total_sales_per_product, 1):
        print(f"Product {i}: ${total:,.2f}")

    print("\n4. Average Monthly Sales by Product:")
    for i, avg in enumerate(avg_monthly_sales_per_product, 1):
        print(f"Product {i}: ${avg:,.2f}")

    print("\n5. Total Sales by Month:")
    for i, total in enumerate(total_sales_per_month, 1):
        print(f"Month {i}: ${total:,.2f}")

    print("\n6. Average Sales Across All Products by Month:")
    for i, avg in enumerate(avg_sales_per_month, 1):
        print(f"Month {i}: ${avg:,.2f}")

    print("\n=== Summary Statistics ===")
    print(f"Total Products: {num_products}")
    print(f"Total Months: {num_months}")
    print(f"Grand Total Sales: ${np.sum(sales_data):,.2f}")
    print(f"Overall Monthly Average: ${np.mean(sales_data):,.2f}")


if __name__ == "__main__":
    # Create sample data files if they don't exist
    if not all(os.path.exists(f) for f in ['sales_data_q1.txt', 'sales_data_q2.txt', 'sales_data_q3.txt']):
        create_sample_data()

    # List of input files
    files = ['sales_data_q1.txt', 'sales_data_q2.txt', 'sales_data_q3.txt']

    # Merge quarterly data
    sales_data = merge_quarterly_data(files)

    # Calculate statistics
    stats = calculate_statistics(sales_data)

    # Print results
    print_results(sales_data, *stats)