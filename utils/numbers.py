import math
from datetime import datetime


def value_exceeded_by_n_percent(ascending_sorted_list: list, n: float) -> float:
    """
    Calculates the value that a given percentage of the items are larger than.

    Args:
        ascending_sorted_list: A list of numbers sorted in ascending order.
        n: The percentage of items that are larger than the returned value (from 0 to 100).

    Returns:
        The value from the list that n percent of the items are larger than.
    """
    return value_below_n_percent(ascending_sorted_list, 100 - n)


def value_below_n_percent(ascending_sorted_list: list, n: float) -> float:
    """
    Calculates the value below which a given percentage of the data falls.

    This function uses linear interpolation to find the percentile value.

    Args:
        ascending_sorted_list: A list of numbers sorted in ascending order.
        n: The percentage (from 0 to 100).

    Returns:
        The value from the list below which n percent of the data is found.
    """
    if not 0 <= n <= 100:
        raise ValueError("Percentage 'n' must be between 0 and 100.")
    if not ascending_sorted_list:
        raise ValueError("Input list cannot be empty.")

    # Calculate the rank of the percentile. [2, 10, 13]
    rank = (n / 100) * (len(ascending_sorted_list) - 1)

    # If the rank is an integer, the value is at that index.
    if rank.is_integer():
        return ascending_sorted_list[int(rank)]

    # If the rank is not an integer, we need to interpolate.
    else:
        lower_index = math.floor(rank)
        upper_index = math.ceil(rank)
        fraction = rank - lower_index

        lower_value = ascending_sorted_list[lower_index]
        upper_value = ascending_sorted_list[upper_index]

        return lower_value + (upper_value - lower_value) * fraction


def to_ms(minutes=0, seconds=0):
    return minutes * 60 * 1000 + seconds * 1000


def timestamp_of(year, month, day):
    return int(datetime(year, month, day).timestamp() * 1000)


def to_timestamp(dt: datetime):
    return int(dt.timestamp() * 1000)


def as_approx_ratio(a, b):
    both = a, b
    min_both = min(both)
    if min_both == 0:
        return f"{a:.0f} : {b:.0f}"
    return " : ".join(f'{b // min_both:.0f}' for b in both)
