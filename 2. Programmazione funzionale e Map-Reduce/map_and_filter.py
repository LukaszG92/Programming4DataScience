from typing import List, Iterator, Any


def square_and_filter_even(numbers: List[int]) -> List[int]:
    squared: Iterator[int] = map(lambda x: x * x, numbers)

    return list(filter(lambda x: (int(x ** 0.5)) % 2 == 0, squared))


def double_and_filter_odd(numbers: List[int]) -> List[int]:
    doubled: Iterator[int] = map(lambda x: x * 2, numbers)

    return list(filter(lambda x: (x // 2) % 2 == 1, doubled))


def cube_and_filter_positive(numbers: List[int]) -> List[int]:
    cubed: Iterator[int] = map(lambda x: x ** 3, numbers)

    return list(filter(lambda x: x > 0, cubed))


if __name__ == "__main__":
    # Test case 1: Basic test with positive numbers
    test_list1: List[int] = [1, 2, 3, 4, 5]
    print("Test 1 - Basic test with positive numbers:")
    print(f"Input: {test_list1}")
    print(f"square_and_filter_even: {square_and_filter_even(test_list1)}")
    print(f"double_and_filter_odd: {double_and_filter_odd(test_list1)}")
    print(f"cube_and_filter_positive: {cube_and_filter_positive(test_list1)}\n")

    # Test case 2: Test with negative numbers
    test_list2: List[int] = [-3, -2, -1, 0, 1, 2, 3]
    print("Test 2 - Test with negative numbers:")
    print(f"Input: {test_list2}")
    print(f"square_and_filter_even: {square_and_filter_even(test_list2)}")
    print(f"double_and_filter_odd: {double_and_filter_odd(test_list2)}")
    print(f"cube_and_filter_positive: {cube_and_filter_positive(test_list2)}\n")

    # Test case 3: Empty list
    test_list3: List[int] = []
    print("Test 3 - Empty list:")
    print(f"Input: {test_list3}")
    print(f"square_and_filter_even: {square_and_filter_even(test_list3)}")
    print(f"double_and_filter_odd: {double_and_filter_odd(test_list3)}")
    print(f"cube_and_filter_positive: {cube_and_filter_positive(test_list3)}")