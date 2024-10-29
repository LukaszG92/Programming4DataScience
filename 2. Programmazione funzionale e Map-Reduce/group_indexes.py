from functools import reduce
from typing import List, Dict, Any


def group_indexes(numbers: List[Any]) -> Dict[Any, List[int]]:
    indexed_pairs = map(lambda x: (x[1], x[0]), enumerate(numbers))

    value_index_pairs = list(map(lambda x: {x[0]: [x[1]]}, indexed_pairs))

    def reducer(acc: Dict[Any, List[int]], curr: Dict[Any, List[int]]) -> Dict[Any, List[int]]:
        for value, indexes in curr.items():
            if value in acc:
                acc[value].extend(indexes)
            else:
                acc[value] = indexes
        return acc

    return reduce(reducer, value_index_pairs, {})


if __name__ == "__main__":

    # Example usage
    sample_lists = [
        # Original example
        [12, 34, 35, 12, 99, 34, 34, 12, 34, 12],
        # All same values
        [5, 5, 5, 5, 5],
        # Mixed types
        ["x", "y", "x", 1, "y", 1],
        # Single value
        [42],
        # Empty list
        []
    ]

    print("\nExample results:")
    for i, sample in enumerate(sample_lists, 1):
        result = group_indexes(sample)
        print(f"\nExample {i}:")
        print(f"Input:  {sample}")
        print(f"Output: {result}")