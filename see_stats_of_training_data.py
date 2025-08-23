import csv
import sys
from statistics import median


def main(filename):
    # These are the columns we want stats on:
    stats_cols = ['pp1', 'pp2', 'pp3', 'pp4', 'ov1', 'ov2', 'ov3', 'ov4']

    # Prepare storage for values:
    values = {col: [] for col in stats_cols}
    is_mod_values = {col: [] for col in stats_cols}

    # Counters for prev_is_mod
    count_mod = 0
    count_not_mod = 0

    with open(filename, newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        # Check that required columns exist
        for col in stats_cols + ['prev_is_mod']:
            if col not in reader.fieldnames:
                print(f"Error: column '{col}' not found in the file.", file=sys.stderr)
                return

        # Read rows
        for row in reader:
            # collect numeric columns
            for col in stats_cols:
                try:
                    val = float(row[col])
                    values[col].append(val)
                    if row['is_mod'].strip() == '1':
                        is_mod_values[col].append(val)
                except ValueError:
                    # skip or handle missing/non-numeric
                    pass

            # tally prev_is_mod
            pim = row['prev_is_mod'].strip()
            if pim == '1':
                count_mod += 1
            elif pim == '0':
                count_not_mod += 1
            # else ignore other values

    # Compute and print stats
    print("Column   Max        Median     Average")
    print("----------------------------------------")
    for col in stats_cols:
        lst = values[col]
        if not lst:
            print(f"{col:6s}   (no data)")
            continue
        print_stats(col, lst)

        # Compute and print stats
    print("\nin the case of is_mod = True")
    print("Column   Max        Median     Average")
    print("----------------------------------------")
    for col in stats_cols:
        lst = is_mod_values[col]
        if not lst:
            print(f"{col:6s}   (no data)")
            continue
        print_stats(col, lst)

    # Print counts
    print("\nCounts of prev_is_mod:")
    print(f"  1 : {count_mod}")
    print(f"  0 : {count_not_mod}")
    print(count_mod / count_not_mod)


def print_stats(col, lst):
    mx = max(lst)
    med = median(lst)
    avg = sum(lst) / len(lst)
    print(f"{col:6s}   {mx:10.4f}   {med:10.4f}   {avg:10.4f}")


if __name__ == '__main__':
    main(r"training_data.csv")
