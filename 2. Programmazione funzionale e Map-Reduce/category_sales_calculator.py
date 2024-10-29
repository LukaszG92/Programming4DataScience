from functools import reduce
from typing import List, Dict, Union


def calculate_category_sales(sales_data: List[Dict[str, Union[str, float, int]]]) -> Dict[str, float]:
    mapped_sales = map(
        lambda sale: {
            sale["category"]: round(float(sale["amount"]), 2)
        },
        sales_data
    )

    def reducer(acc: Dict[str, float], curr: Dict[str, float]) -> Dict[str, float]:
        for category, amount in curr.items():
            acc[category] = round(acc.get(category, 0) + amount, 2)
        return acc

    return reduce(reducer, mapped_sales, {})


def get_category_stats(category_sales: Dict[str, float]) -> Dict[str, Union[float, str]]:
    if not category_sales:
        return {
            "total_sales": 0.0,
            "average_category_sales": 0.0,
            "top_category": "None",
            "number_of_categories": 0
        }

    total_sales = round(sum(category_sales.values()), 2)
    avg_sales = round(total_sales / len(category_sales), 2)
    top_category = max(category_sales.items(), key=lambda x: x[1])[0]

    return {
        "total_sales": total_sales,
        "average_category_sales": avg_sales,
        "top_category": top_category,
        "number_of_categories": len(category_sales)
    }


if __name__ == "__main__":
    # Example usage with sample data
    sample_sales_data = [
        {"id": 1, "category": "Electronics", "amount": 1299.99, "date": "2024-01-01"},
        {"id": 2, "category": "Books", "amount": 49.99, "date": "2024-01-01"},
        {"id": 3, "category": "Electronics", "amount": 799.50, "date": "2024-01-02"},
        {"id": 4, "category": "Clothing", "amount": 159.99, "date": "2024-01-02"},
        {"id": 5, "category": "Books", "amount": 29.99, "date": "2024-01-03"},
        {"id": 6, "category": "Electronics", "amount": 499.99, "date": "2024-01-03"}
    ]

    # Calculate sales by category
    category_sales = calculate_category_sales(sample_sales_data)
    stats = get_category_stats(category_sales)

    print("\nSales by Category:")
    for category, total in category_sales.items():
        print(f"{category}: ${total:,.2f}")

    print("\nSales Statistics:")
    print(f"Total Sales: ${stats['total_sales']:,.2f}")
    print(f"Average Category Sales: ${stats['average_category_sales']:,.2f}")
    print(f"Top Performing Category: {stats['top_category']}")
    print(f"Number of Categories: {stats['number_of_categories']}")