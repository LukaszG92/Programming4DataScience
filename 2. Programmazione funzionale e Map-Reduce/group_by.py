from functools import reduce
from typing import List, Tuple, Dict


def group_and_sum(data: List[Tuple[str, int]]) -> Dict[str, int]:
    mapped_data = map(lambda x: {x[0]: x[1]}, data)

    def reducer(acc: Dict[str, int], curr: Dict[str, int]) -> Dict[str, int]:
        for key, value in curr.items():
            acc[key] = acc.get(key, 0) + value
        return acc

    return reduce(reducer, mapped_data, {})


if __name__ == "__main__":
    # Example usage
    data = [
        ("fruits", 10),
        ("vegetables", 20),
        ("fruits", 30),
        ("dairy", 15),
        ("vegetables", 10)
    ]

    result = group_and_sum(data)
    print("\nExample usage result:")
    for key, value in result.items():
        print(f"{key}: {value}")