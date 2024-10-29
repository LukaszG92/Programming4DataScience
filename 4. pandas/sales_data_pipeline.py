import pandas as pd
from typing import Dict


def create_sample_data() -> None:
    data = {
        'Date': [
            '2024-01-15', '2024-01-20', '2024-01-25', '2024-02-01', '2024-02-05',
            '2024-02-10', '2024-02-15', '2024-03-01', '2024-03-05', '2024-03-10',
            '2024-03-15', '2024-03-20', '2024-03-25', '2024-03-30', '2024-04-01'
        ],
        'Product_Name': [
            'Laptop', 'T-shirt', 'Office Chair', 'Smartphone', 'Desk',
            'Tablet', 'Jeans', 'Bookshelf', 'Headphones', 'Sofa',
            'Monitor', 'Dress', 'Coffee Table', 'Smart Watch', 'Filing Cabinet'
        ],
        'Quantity': [
            3, 10, 2, 5, 1,
            4, 8, 2, 6, 1,
            2, 5, 3, 4, 2
        ],
        'Price': [
            1200.00, 25.99, 299.99, 799.99, 399.99,
            599.99, 49.99, 199.99, 149.99, 899.99,
            349.99, 79.99, 249.99, 299.99, 199.99
        ]
    }

    df = pd.DataFrame(data)
    df.to_csv('sales_data.csv', index=False)
    print("Sample data created and saved to 'sales_data.csv'")


def load_and_inspect_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)

    print("\n=== Data Overview ===")
    print("\nFirst few rows:")
    print(df.head())

    print("\nDataset Info:")
    print(df.info())

    print("\nBasic Statistics:")
    print(df.describe())

    return df


def categorize_product(product_name: str) -> str:
    categories = {
        'Electronics': ['Laptop', 'Smartphone', 'Tablet', 'Headphones', 'Monitor', 'Smart Watch'],
        'Clothing': ['T-shirt', 'Jeans', 'Dress'],
        'Furniture': ['Office Chair', 'Desk', 'Bookshelf', 'Sofa', 'Coffee Table', 'Filing Cabinet']
    }

    for category, products in categories.items():
        if any(product.lower() in product_name.lower() for product in products):
            return category

    return 'Other'


def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    transformed_df = df.copy()

    transformed_df['Total_Sales'] = transformed_df['Quantity'] * transformed_df['Price']

    transformed_df['Date'] = pd.to_datetime(transformed_df['Date'])
    transformed_df['Month'] = transformed_df['Date'].dt.strftime('%B')

    transformed_df['Category'] = transformed_df['Product_Name'].apply(categorize_product)

    print("\n=== Transformed Data ===")
    print("\nFirst few rows of transformed data:")
    print(transformed_df.head())

    return transformed_df


def calculate_statistics(df: pd.DataFrame) -> Dict[str, float]:
    stats = {
        'Mean': df['Total_Sales'].mean(),
        'Median': df['Total_Sales'].median(),
        'Standard Deviation': df['Total_Sales'].std(),
        'Maximum': df['Total_Sales'].max(),
        'Minimum': df['Total_Sales'].min(),
        'Total Revenue': df['Total_Sales'].sum()
    }

    print("\n=== Summary Statistics for Total Sales ===")
    for stat, value in stats.items():
        print(f"{stat}: ${value:,.2f}")

    return stats


def analyze_sales_by_category(df: pd.DataFrame) -> None:
    print("\n=== Sales Analysis by Category ===")

    category_sales = df.groupby('Category')['Total_Sales'].agg(['sum', 'mean', 'count'])
    category_sales = category_sales.round(2)

    print("\nCategory-wise Sales Summary:")
    print(category_sales)

    total_sales = df['Total_Sales'].sum()
    category_percentages = (category_sales['sum'] / total_sales * 100).round(2)

    print("\nPercentage of Total Sales by Category:")
    for category, percentage in category_percentages.items():
        print(f"{category}: {percentage}%")


def analyze_monthly_trends(df: pd.DataFrame) -> None:
    print("\n=== Monthly Sales Analysis ===")

    monthly_sales = df.groupby('Month')['Total_Sales'].agg(['sum', 'mean', 'count'])
    monthly_sales = monthly_sales.round(2)

    print("\nMonthly Sales Summary:")
    print(monthly_sales)


if __name__ == "__main__":
    file_path = 'sales_data.csv'

    try:
        pd.read_csv(file_path)
    except FileNotFoundError:
        create_sample_data()

    df = load_and_inspect_data(file_path)

    transformed_df = transform_data(df)

    stats = calculate_statistics(transformed_df)

    analyze_sales_by_category(transformed_df)
    analyze_monthly_trends(transformed_df)