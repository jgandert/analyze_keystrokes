import csv
from pathlib import Path

from utils.constants import VK_CODE_TO_LETTER
from utils.numbers import to_ms
from utils.serialization import write_to_csv_gz, fix_csv_field_size

PATH_TO_DATASET = Path(r"dataset2")

fix_csv_field_size()


def preprocess(events):
    # why is it so hard to find a proper keystroke dataset that is not messed up
    # in every possible way (wrong order, up but not down, down but not up, ...)

    result = []
    for event_str in events:
        parts = event_str.strip().split()
        if len(parts) != 3:
            continue

        time_ms_str, event_type, keycode = parts

        if event_type not in ("KeyDown", "KeyUp"):
            continue

        mapped_key = VK_CODE_TO_LETTER.get(keycode)
        if mapped_key is None or not time_ms_str.isdigit():
            continue

        time_ms = int(time_ms_str)
        if time_ms < 0:
            continue

        key = mapped_key.lower().strip().replace(' ', '_')
        is_down = 1 if event_type == 'KeyDown' else 0
        result.append((time_ms, is_down, key))
    result.sort(key=lambda i: i[0])
    return result


def process_review_meta_files(writer) -> None:
    """
    Scans the specified dataset directory for CSV files, processes 'ReviewMeta' rows,
    and writes the transformed event data to the provided writer, ignoring OS-level
    key repeats.

    Args:
        writer: A CSV writer object to write the output rows.
    """
    csv_files = list(PATH_TO_DATASET.rglob('*.csv'))

    if not csv_files:
        print(f"No CSV files found in the dataset directory.")
        return

    last_t = 0  # This holds the timestamp of the last written event or marker.

    for file_path in csv_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter='\t')

                try:
                    header = next(reader)
                except StopIteration:
                    continue  # Skip empty files

                try:
                    review_meta_index = header.index('ReviewMeta')
                except ValueError:
                    print(f"Warning: 'ReviewMeta' header not found in {file_path.name}. Skipping.")
                    continue

                process_reader(last_t, reader, review_meta_index, writer)
        except IOError as e:
            print(f"Error reading file {file_path.name}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred while processing {file_path.name}: {e}")


def process_reader(last_t, reader, review_meta_index, writer):
    for row in reader:
        if len(row) <= review_meta_index:
            continue

        review_meta_data = row[review_meta_index]
        if not review_meta_data:
            continue

        events = preprocess(review_meta_data.split(';'))
        last_t = process_events(events, last_t, writer)
    return last_t


def process_events(events, last_t, writer):
    # This set tracks keys held down to prevent repeats for the current row.
    keys_currently_down_to_time = dict()
    processed_events_for_row = []
    first_event_time_in_row = -1

    for time_ms, is_down, key in events:
        if not is_down and (key not in keys_currently_down_to_time or (
                time_ms - keys_currently_down_to_time[key]) > 10_000):
            # On error, discard previously collected events for this row
            processed_events_for_row.clear()
            # Also reset the key state
            keys_currently_down_to_time.clear()
            first_event_time_in_row = -1
            continue

        if first_event_time_in_row == -1:
            first_event_time_in_row = time_ms

        time_delta = time_ms - first_event_time_in_row
        absolute_time = last_t + time_delta

        # Logic to handle key state and prevent repeats
        if is_down:
            if key in keys_currently_down_to_time:
                continue  # Key is already down, this is a repeat, skip.
            keys_currently_down_to_time[key] = time_ms
        elif not is_down:
            del keys_currently_down_to_time[key]

        processed_events_for_row.append((absolute_time, is_down, key))

    # Check if the remaining items are only key presses or releases that were never pressed
    # If so, we remove them
    while keys_currently_down_to_time:
        _, is_down, key = processed_events_for_row.pop()
        if not is_down or key not in keys_currently_down_to_time:
            # The last event is not down, or does not match one still down.
            return last_t

        del keys_currently_down_to_time[key]

    # Check if anything valid is left
    if processed_events_for_row:
        row_last_event_time = 0
        for abs_time, is_down, key in processed_events_for_row:
            writer.writerow((abs_time, is_down, key))
            row_last_event_time = abs_time

        last_t = row_last_event_time

        # Insert the marker element
        marker_time = last_t + to_ms(seconds=30)
        writer.writerow((marker_time, None, None))

        last_t = marker_time
    return last_t


if __name__ == '__main__':
    write_to_csv_gz(Path() / 'dataset' / 'dataset2.csv.gz', process_review_meta_files)
