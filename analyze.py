import csv
import gzip
import sys
from collections import Counter, defaultdict, ChainMap
from itertools import product
from pathlib import Path
from typing import Iterable

from utils.columns import TrainingDataColId
from utils.constants import TIME_I, IS_DOWN_I, KEY_I, MAIN_AREA_KEYS, MOD_KEYS, is_left_not_right, \
    NON_MOD_VALUE, MOD_VALUE
from utils.format import pct_row, print_table, pct2_row, trim_trailing_zero, print_h1, \
    print_h2, print_h3
from utils.numbers import value_below_n_percent, value_exceeded_by_n_percent, as_approx_ratio
from utils.pattern import PermutationPatternMatcher
from utils.serialization import TabMinDialect, read_from_csv_gz
from utils.training_data import TrainingData

DATASET_PATH = Path(__file__).parent / 'dataset'

LOG_PATH = DATASET_PATH / 'dataset2.csv.gz'
LOG_PATH = DATASET_PATH / 'filtered_events.csv.gz'
TRAIN_PATH = DATASET_PATH / f'training_data.csv.gz'

PRINT_STATISTICS = True
PRINT_INDIVIDUAL = False
DURATION_NUM_FMT = '_.0f'

N_MOST_COMMON = 32

KEYS_PERCENTAGE_BELOW_LOW = 75
KEYS_PERCENTAGE_BELOW_MID = 99.0
KEYS_PERCENTAGE_BELOW_HIGH = 99.9

NUM_ABOVE_HIGH_TXT = f'< {trim_trailing_zero(KEYS_PERCENTAGE_BELOW_HIGH)}%'
NUM_ABOVE_MID_TXT = f'< {trim_trailing_zero(KEYS_PERCENTAGE_BELOW_MID)}%'
NUM_ABOVE_LOW_TXT = f'< {trim_trailing_zero(KEYS_PERCENTAGE_BELOW_LOW)}%'
NUM_BELOW_LOW_TXT = f'{trim_trailing_zero(KEYS_PERCENTAGE_BELOW_LOW)}% <'
NUM_BELOW_MID_TXT = f'{trim_trailing_zero(KEYS_PERCENTAGE_BELOW_MID)}% <'
NUM_BELOW_HIGH_TXT = f'{trim_trailing_zero(KEYS_PERCENTAGE_BELOW_HIGH)}% <'

# does not include removals from key being pressed down consecutively
event_after_this_one_was_removed = set()


def print_stats(numbers, fmt=DURATION_NUM_FMT):
    if not numbers:
        print("was empty\n")
        return

    sorted_numbers = sorted(numbers)
    most_below_low = value_below_n_percent(sorted_numbers, KEYS_PERCENTAGE_BELOW_LOW)
    most_below_mid = value_below_n_percent(sorted_numbers, KEYS_PERCENTAGE_BELOW_MID)
    most_below_high = value_below_n_percent(sorted_numbers, KEYS_PERCENTAGE_BELOW_HIGH)

    most_above_low = value_exceeded_by_n_percent(sorted_numbers, KEYS_PERCENTAGE_BELOW_LOW)
    most_above_mid = value_exceeded_by_n_percent(sorted_numbers, KEYS_PERCENTAGE_BELOW_MID)
    most_above_high = value_exceeded_by_n_percent(sorted_numbers, KEYS_PERCENTAGE_BELOW_HIGH)

    min_value = sorted_numbers[0]
    max_value = sorted_numbers[-1]
    count = len(sorted_numbers)
    total_sum = sum(sorted_numbers)
    avg = total_sum / count

    mid = count // 2
    if count > 1 and count % 2 == 0:
        median = (sorted_numbers[mid - 1] + sorted_numbers[mid]) / 2
    else:
        median = sorted_numbers[mid]

    print_table([
        [
            'min',
            NUM_ABOVE_HIGH_TXT,
            NUM_ABOVE_MID_TXT,
            NUM_ABOVE_LOW_TXT,
            'median',
            'avg',
            NUM_BELOW_LOW_TXT,
            NUM_BELOW_MID_TXT,
            NUM_BELOW_HIGH_TXT,
            'max',
        ],
        None,
        [
            f"{min_value:{fmt}}",
            f"{most_above_high:{fmt}}",
            f"{most_above_mid:{fmt}}",
            f"{most_above_low:{fmt}}",
            f"{median:{fmt}}",
            f"{avg:{fmt}}",
            f"{most_below_low:{fmt}}",
            f"{most_below_mid:{fmt}}",
            f"{most_below_high:{fmt}}",
            f"{max_value:{fmt}}",
        ]
    ], right_align_columns=range(0, 99))
    print()


def print_intersection_stats(intersections, space=None):
    all_overlap_durations = []
    all_overlap_percentages = []
    all_durations_between_both_pressed = []

    space_with_others_overlap_durations = []
    space_duration_between_both_presses = []

    for keys, intersection_data in intersections.items():
        intersection_durations = []
        intersection_percentages = []
        intersections_durations_between_both_presses = []
        for (overlap_duration, overlap_percentage,
             duration_between_both_pressed) in intersection_data:
            intersection_durations.append(overlap_duration)
            if overlap_percentage >= 0:
                intersection_percentages.append(overlap_percentage)
            intersections_durations_between_both_presses.append(duration_between_both_pressed)

        if PRINT_INDIVIDUAL:
            print_h2(f"{keys[0]} with {keys[1]}")
            print_h3("Intersection durations")
            print_stats(intersection_durations)
        all_overlap_durations.extend(intersection_durations)

        if PRINT_INDIVIDUAL:
            print_h3("Intersection percentages")
            print_stats(intersection_percentages, fmt='_.2f')
        all_overlap_percentages.extend(intersection_percentages)

        if PRINT_INDIVIDUAL:
            print_h3("Duration between presses of intersecting keys")
            print_stats(intersections_durations_between_both_presses)
        all_durations_between_both_pressed.extend(intersections_durations_between_both_presses)

        if keys[0] == "space":
            space_with_others_overlap_durations.extend(intersection_durations)
            space_duration_between_both_presses.extend(intersections_durations_between_both_presses)

    if PRINT_INDIVIDUAL:
        print_h2(f"All")
    print_h3("Intersection durations")
    print_stats(all_overlap_durations)

    print_h3("Intersection percentages")
    print_stats(all_overlap_percentages, fmt='_.2f')

    print_h3("Durations between both presses")
    print_stats(all_durations_between_both_pressed)

    if space is None:
        return

    if space_with_others_overlap_durations:
        print_h2("Space first{space}")
        print()
        print_h3("Intersection durations")
        print_stats(space_with_others_overlap_durations)

    if space_duration_between_both_presses:
        print_h3("Durations between both presses")
        print_stats(space_duration_between_both_presses)


LABEL_MAP = {
    MOD_VALUE: "mod",
    NON_MOD_VALUE: "non-mod",
}


def heading_for_pair(first=None, second=None):
    if second is None and first is None:
        raise ValueError("At least one of first or second must be provided")

    if second is None:
        return f"{LABEL_MAP[first]} keys first"
    if first is None:
        return f"{LABEL_MAP[second]} keys second"
    return f"{LABEL_MAP[first]} keys first, {LABEL_MAP[second]} keys second"


def print_intersections_individual(intersections):
    for first, second in product((NON_MOD_VALUE, MOD_VALUE), repeat=2):
        heading = heading_for_pair(first, second)
        print_h2(heading)
        print_intersection_stats(intersections[first, second])


def print_intersections_combined_by_first(intersections):
    for first in (NON_MOD_VALUE, MOD_VALUE):
        cm = ChainMap(
            intersections[first, NON_MOD_VALUE],
            intersections[first, MOD_VALUE],
        )

        print_h2(heading_for_pair(first))
        print_intersection_stats(cm)


def print_intersections_combined_by_second(intersections):
    for second in (NON_MOD_VALUE, MOD_VALUE):
        cm = ChainMap(
            intersections[NON_MOD_VALUE, second],
            intersections[MOD_VALUE, second],
        )

        h = heading_for_pair(second=second)
        print_h2(h)
        print_intersection_stats(cm, space=f', {h}')


def print_intersections_all_reports(intersections):
    print_intersection_stats(ChainMap(*intersections.values()), space='')  # all combined
    print_intersections_combined_by_first(intersections)
    print_intersections_combined_by_second(intersections)
    print_intersections_individual(intersections)


def load_events(path: Path):
    return read_from_csv_gz(path, map_keys_and_remove_pause_marker)


def map_keys_and_remove_pause_marker(
        rows: Iterable[tuple[str, str, str]]
) -> list[tuple[int, bool, str]]:
    result = []
    prev_row = None

    for timestamp, is_down, key in rows:
        if key == '':
            event_after_this_one_was_removed.add(prev_row)
            continue

        timestamp = int(timestamp)
        is_down = is_down == '1'

        if key is None:
            # This doesn't apply at all for most datasets:
            # The "," is most likely incorrect, but we don't know the actual
            # values because it was before the pynput vk change
            # (from now on we will get an int instead of None instead).
            key = ","
        else:
            if isinstance(key, int):
                print("don't know what this is (skipping):", key, timestamp)
                event_after_this_one_was_removed.add(prev_row)
                continue

            # this can boost performance by making string comparisons O(1) as only the identity is compared
            # it also saves memory by not repeating same strings in memory
            key = sys.intern(key.lower())

        row = (timestamp, is_down, key)
        result.append(row)
        prev_row = row

    return result


def p_rows(counter: Counter, postfix=""):
    total = counter.total()
    if not postfix.startswith(" "):
        postfix = " " + postfix
    return [
        pct_row("non-mod" + postfix, counter[NON_MOD_VALUE], total),
        pct_row("mod" + postfix, counter[MOD_VALUE], total),
        ["total" + postfix, total],
    ]


def p_multi_rows(counter: Counter, interfix: str):
    total = counter.total()
    return [
        pct_row("non-mod " + interfix + " non-mod", counter[(NON_MOD_VALUE, NON_MOD_VALUE)], total),
        pct_row("non-mod " + interfix + " mod", counter[(NON_MOD_VALUE, MOD_VALUE)], total),
        pct_row("mod " + interfix + " non-mod", counter[(MOD_VALUE, NON_MOD_VALUE)], total),
        pct_row("mod " + interfix + " mod", counter[(MOD_VALUE, MOD_VALUE)], total),
        ["total " + interfix, total],
    ]


def count_any_first(counter: Counter, first_key):
    return counter[(first_key, NON_MOD_VALUE)] + counter[(first_key, MOD_VALUE)]


def count_any_second(counter: Counter, second_key):
    return counter[(NON_MOD_VALUE, second_key)] + counter[(MOD_VALUE, second_key)]


def count_any(key, *multiple_counter):
    result = 0
    for counter in multiple_counter:
        for keys, count in counter.items():
            if key in keys:
                result += count
    return result


def p_variations(key, zero_overlap_counts, overlap_counts, wrap_counts):
    opposite_key = not key
    opposite_name = "mod" if opposite_key else "non-mod"

    # will be different from the version from total_counts, because wrap and overlap have two
    # elements, and so will be counted multiple times here
    total = zero_overlap_counts[key] + count_any(key, overlap_counts, wrap_counts)
    return [
        ['type', 'count', '%'],
        None,
        pct_row("no overlap/wrap", zero_overlap_counts[key], total),
        [],
        pct_row("any overlap", count_any(key, overlap_counts), total),
        pct_row("- overlaps non-mod", overlap_counts[(key, NON_MOD_VALUE)], total),
        pct_row("- overlaps mod", overlap_counts[(key, MOD_VALUE)], total),
        pct_row("- overlapped by " + opposite_name, overlap_counts[(opposite_key, key)], total),
        [],
        pct_row("any wrap", count_any(key, wrap_counts), total),
        pct_row("- wraps non-mod", wrap_counts[(key, NON_MOD_VALUE)], total),
        pct_row("- wraps mod", wrap_counts[(key, MOD_VALUE)], total),
        pct_row("- wrapped by " + opposite_name, wrap_counts[(opposite_key, key)], total),
    ]


def print_counter_as_table(counter, header, n_most_common):
    l = [header, None]
    for k, v in counter.most_common(n_most_common):
        l.append([k, v])
    print_table(l)


# a = originally added to not exclude overlaps with p, but might be the actual prev
# p = previous, x = tap or hold key, n = second (after tap-hold), t = third after x and n
KEYSTROKES_PATTERN = 'ApAPAxPnPtXtNtXt'

# make sure this is consistent with TrainingDataColId and InputTrainCol
KEYSTROKES_ORDER = 'xnNXt'


def collect_training_data_by_pattern(events, mod_counter: Counter, ev_to_pp_ov: dict):
    result = TrainingData([])

    matcher = PermutationPatternMatcher(KEYSTROKES_PATTERN, down_index=IS_DOWN_I, key_index=KEY_I,
                                        released_letters_to_ignore=['t'])
    for match in matcher.all_matches(events, continue_at_letter='x'):
        th_down = events[match['x']]

        is_mod = th_down[KEY_I] in MOD_KEYS

        # If we didn't do this, the times would be skewed due to non-main-area
        # keys being farther away than main ones on traditional keyboards.
        next_down = events[match['n']]
        is_next_main = next_down[KEY_I] in MAIN_AREA_KEYS
        if is_mod and not is_next_main:
            continue

        # After 700 ms everybody will expect hold, so it's not useful here.
        # Same for third down:
        # It can't be relevant if the tap-hold has been pressed so long.
        has_third = match['t'] < match['X']
        third_down = events[match['t']]
        if ((next_down[TIME_I] - th_down[TIME_I]) > 700 or
                (has_third and (third_down[TIME_I] - th_down[TIME_I]) > 700)):
            continue

        last = max(match.values())
        all_but_last = (events[j] for j in match.values() if j != last)
        if any(e in event_after_this_one_was_removed for e in all_but_last):
            continue

        key_pressed_before_pth = events[match['P']]
        key_pressed_before_pth_is_mod = key_pressed_before_pth[KEY_I] in MOD_KEYS

        # if P (release of p) is after x (PTH), then choose A (it's always before x)
        if match['x'] < match['P']:
            key_released_before_pth = events[match['A']]
        else:
            key_released_before_pth = key_pressed_before_pth

        key_released_before_pth_is_mod = key_released_before_pth[KEY_I] in MOD_KEYS

        second_is_mod = next_down[KEY_I] in MOD_KEYS
        r = [is_mod, second_is_mod, key_pressed_before_pth_is_mod, key_pressed_before_pth[TIME_I],
             key_released_before_pth_is_mod, key_released_before_pth[TIME_I]]

        for o in KEYSTROKES_ORDER:
            r.append(events[match[o]][TIME_I])

        pp_ov = ev_to_pp_ov.pop(th_down, None)
        if pp_ov is None:
            continue
        r.extend(pp_ov)

        pth_left = is_left_not_right(th_down[KEY_I])
        pth_and_second_same_side_but_third_not = (pth_left == is_left_not_right(
            next_down[KEY_I]) and pth_left != is_left_not_right(third_down[KEY_I]))

        # if you want ALL the training data
        # r.append(pth_and_second_same_side_but_third_not)

        result.elements.append(tuple(r))

        # Here is the problem: The training data is derived from a typing test in which participants
        # type sentences. Most of these start with an uppercase letter, thus requiring Shift.
        # As a result, the data is skewed so that empty press-to-press values (-1) often correlate
        # with a mod activation. Search for 'event_after_this_one_was_removed' for more.
        # To avoid this issue, we simply have to remove training data where the last pp are -1.
        if pp_ov[KEPT_HISTORY_SIZE - 1] < 0:
            continue

        is_triple_down = match['n'] + 1 == match['t']
        is_wrapped = match['N'] < match['X']
        result.add_count(is_mod, is_wrapped, is_triple_down)
        if is_mod:
            mod_counter[th_down[KEY_I]] += 1

    return result


"""
Explanation of the following data extraction:

lower case letter = press
upper case letter = release
x = tap-hold press

press to press recorded
|    |   | |            |
a  A
     b          B
         c  C
           d         D
                        x
   |        |   |    |
overlap recorded
"""
HISTORY_SIZE = 4

# if you want ALL the training data, both HISTORY_SIZE have to be equal
KEPT_HISTORY_SIZE = 2
DEFAULT_DUR = -1
TRAIN_STATS = defaultdict(list)
DEFAULT_NON_MOD_AVG = 102
DEFAULT_MOD_AVG = 229
DEFAULT_OVERLAP_AVG = 32

# The time window (in ms) to look back for recent key releases.
# A key released more than 1s ago is considered "stale".
STALE_THRESHOLD_MS = 1000


def get_recent_duration_avg(press_time, release_history):
    is_mod_as_num_total = 0
    total = 0
    count = 0

    for duration, release_time, is_mod_as_num in release_history:
        time_delta = press_time - release_time

        # Check if the release happened within our time window *before* the press
        if 0 < time_delta < STALE_THRESHOLD_MS:
            is_mod_as_num_total += is_mod_as_num
            total += duration
            count += 1

    if not count:
        return DEFAULT_NON_MOD_AVG, 0.0

    return total / count, is_mod_as_num_total / count


def collect_pp_ov_training_data(events) -> dict:
    result = dict()
    press_to_press_durs = [DEFAULT_DUR] * HISTORY_SIZE
    overlap_durs = [0] * HISTORY_SIZE

    # Indices for the circular history arrays
    press_to_press_index = 0
    overlap_index = 0

    # Counter for currently pressed keys, starts at 0
    down_count = 0

    # Timers to track start times for duration calculations
    press_to_press_timer_start = -1
    overlap_timer_start = -1
    just_now_reset = True

    # ------------ for running average ------------
    non_mod_count = 1
    non_mod_dur_avg = DEFAULT_NON_MOD_AVG
    mod_count = 1
    mod_dur_avg = DEFAULT_MOD_AVG
    overlap_count = 1
    overlap_dur_avg = DEFAULT_OVERLAP_AVG
    first_press_after_release = -1

    keys_down = {}  # Stores {key_code: press_time} for currently held keys
    release_history = []
    keys_down_when = []
    # ---------------------------------------------

    for i, event in enumerate(events):
        current_time = event[TIME_I]
        is_pressed = event[IS_DOWN_I]
        key = event[KEY_I]

        if is_pressed:
            # ------------ for running average ------------
            # When another key is pressed, while we're still tracking one, we
            # will ignore that new key. Yes, this is not perfect, but the goal
            # here is to create training data representative of the real world
            # data that we will capture when code is running in C.
            if first_press_after_release < 0:
                first_press_after_release = current_time
            keys_down[key] = current_time
            keys_down_when.append(current_time)
            # ---------------------------------------------

            if press_to_press_timer_start >= 0:
                # Calculate duration from the previous press
                p_to_p_dur = current_time - press_to_press_timer_start

                # Store duration in the circular buffer
                press_to_press_durs[press_to_press_index] = p_to_p_dur
                press_to_press_index = (press_to_press_index + 1) % HISTORY_SIZE

            # Reset the press-to-press timer for the next press
            press_to_press_timer_start = current_time

            # =================================================================
            # Save the press to press and overlap durations
            # the oldest entry is the one the index points to
            # the last = newest element is actually the index - 1
            pp = wrap_around_list_from(press_to_press_durs, press_to_press_index)
            ov = wrap_around_list_from(overlap_durs, overlap_index)

            # case 1: no down - nothing should be done
            # case 2: 1 down - append 0
            # case 3: 2 down - append 0, then append (cur - ov_start)
            has_overlap = down_count > 1
            for _ in range(down_count - (1 if has_overlap else 0)):
                # no overlap, but must add 0 for consistency - explanation:
                # in the following, x is the current key (lower case = down, upper = up)
                # aAbBcCdDx -> would provide 4 overlaps
                # aAbBcCdx (d not released) -> would provide 3 values without adding 0
                # for press to press durations, we would get 4 in either case
                # if there are even more keys not released yet, we have to do this for each
                ov.append(0)
                del ov[0]

            if has_overlap:
                assert overlap_timer_start >= 0
                ov.append(current_time - overlap_timer_start)
                del ov[0]

            pressed_count_last_400_ms = 0
            for i in range(len(keys_down_when) - 1, -1, -1):
                if current_time - keys_down_when[i] < 400:
                    pressed_count_last_400_ms += 1
                else:
                    # delete too old
                    del keys_down_when[i]

            recent_press_dur_avg, recent_is_mod_avg = get_recent_duration_avg(current_time,
                                                                              release_history)
            '''
            # if you want ALL the training data
            result[event] = (*pp, *ov, down_count, non_mod_dur_avg, mod_dur_avg, overlap_dur_avg,
                             recent_press_dur_avg, recent_is_mod_avg, pressed_count_last_400_ms)
            '''
            result[event] = (pp[2], pp[3], ov[2], ov[3], down_count)

            if not just_now_reset:
                TRAIN_STATS['Count of presses in last 400 ms'].append(pressed_count_last_400_ms)
                TRAIN_STATS['Average of recent press durations'].append(recent_press_dur_avg)
                TRAIN_STATS[
                    'Average of recent is-mod-ness in percent (0 = non-mod, 100 = mod)'].append(
                    100 * recent_is_mod_avg)
                TRAIN_STATS['Average of approx. non-mod durations'].append(non_mod_dur_avg)
                TRAIN_STATS['Average of approx. mod durations'].append(mod_dur_avg)
                TRAIN_STATS['Exponentially moving average of overlap durations'].append(
                    overlap_dur_avg)
                if ov[-1] > 0:
                    TRAIN_STATS[f'Overlap durations > 0'].append(ov[-1])
                TRAIN_STATS[f'Press-to-press durations'].append(pp[-1])

            # =================================================================

            # Must be done after adding as we can't count the current key down,
            # and thus have an overlap, as it is just right now pressed down.
            down_count += 1
            if down_count == 2:
                # Two keys are down at the same time, starting an overlap
                overlap_timer_start = current_time
            just_now_reset = False
        else:  # on release
            # =====================================================================
            # ------------ for running average ------------
            is_mod = key in MOD_KEYS
            if key in keys_down:
                press_time = keys_down.pop(key)
                duration = current_time - press_time

                # Add the release info to our history buffer
                release_history.append((duration, current_time, 1.0 if is_mod else 0.0))
                if len(release_history) > 3:
                    del release_history[0]

            if is_mod:
                if first_press_after_release >= 0:
                    if mod_count < 255:
                        mod_count += 1
                    d = (current_time - first_press_after_release)
                    mod_dur_avg = (d + mod_dur_avg * (mod_count - 1)) / mod_count
            else:
                if first_press_after_release >= 0:
                    if non_mod_count < 255:
                        non_mod_count += 1
                    d = (current_time - first_press_after_release)
                    non_mod_dur_avg = (d + non_mod_dur_avg * (non_mod_count - 1)) / non_mod_count
            first_press_after_release = -1
            # ---------------------------------------------

            overlap = 0

            # Check if an overlap was active (i.e., at least 2 keys are down before this release)
            if down_count >= 2:
                overlap = current_time - overlap_timer_start
                if overlap > 0:
                    '''
                    if overlap_count < 255:
                        # yes, after the max is reached, we just pretend we've only seen
                        # max elements so far
                        overlap_count += 1
                    # moving average
                    overlap_dur_avg = (overlap + overlap_dur_avg * (overlap_count - 1)) / overlap_count
                    '''

                    if overlap_count < 31:
                        overlap_count += 1
                    # exponentially weighted moving average approximation
                    overlap_dur_avg -= overlap_dur_avg / overlap_count
                    overlap_dur_avg += overlap / overlap_count

            if down_count > 0:
                down_count -= 1

            # Store the calculated overlap (can be 0) in the circular buffer
            overlap_durs[overlap_index] = overlap
            overlap_index = (overlap_index + 1) % HISTORY_SIZE

            # We don't want to count overlaps twice, so we set to the current time
            overlap_timer_start = current_time

        # reset if different user (as durations would not be real)
        if event in event_after_this_one_was_removed:
            # index doesn't matter (have to take care of wrap around anyway)
            # overlap_timer_start doesn't matter, as will be set again
            down_count = 0
            press_to_press_timer_start = -1
            overlap_timer_start = -1
            for j in range(HISTORY_SIZE):
                press_to_press_durs[j] = DEFAULT_DUR
                overlap_durs[j] = 0

            # ------------ for running average ------------
            non_mod_count = 1
            non_mod_dur_avg = DEFAULT_NON_MOD_AVG
            mod_count = 1
            mod_dur_avg = DEFAULT_MOD_AVG
            overlap_count = 1
            overlap_dur_avg = DEFAULT_OVERLAP_AVG
            first_press_after_release = -1

            release_history.clear()
            keys_down.clear()
            keys_down_when.clear()
            just_now_reset = True
            # ---------------------------------------------

    return result


def wrap_around_list_from(lst, index):
    return [lst[j % HISTORY_SIZE] for j in range(index, index + HISTORY_SIZE)]


def find_prev_event(events, start_index, key_to_find, key_to_find_down, max_search_window=7):
    for i in range(start_index, max(0, start_index - max_search_window), -1):
        if events[i][KEY_I] == key_to_find:
            if events[i][IS_DOWN_I] == key_to_find_down:
                return events[i]
            else:
                # we are assuming that we don't want the key we're looking for in the opposite state
                break
    return None


def write_csv_of_training_data(training_data: TrainingData):
    with gzip.open(str(TRAIN_PATH), 'wt', newline='', encoding='utf-8') as gz_file:
        writer = csv.writer(gz_file, dialect=TabMinDialect)

        writer.writerow(TrainingDataColId.member_names)
        for el in training_data.elements:
            writer.writerow((col.to_csv(el[col.value]) for col in TrainingDataColId))

    print(f'Written training data to {TRAIN_PATH}')


def print_counts_vs_with_ratio(total_counts, name, mods_count, non_mods_count):
    print_h1(f'Counts of mods vs non-mods {name} another key')
    print_table([
        ["mods", "non-mods", "ratio"],
        [mods_count, non_mods_count, as_approx_ratio(mods_count, non_mods_count)],
    ], right_align_columns=[0, 1, 2])
    print()
    print('When divided by the total count in each category:')
    print()

    mods_scaled = 100 * mods_count / total_counts[MOD_VALUE]
    non_mods_scaled = 100 * non_mods_count / total_counts[NON_MOD_VALUE]
    print_table([
        ["mods %", "non-mods %", "ratio"],
        [mods_scaled, non_mods_scaled, as_approx_ratio(mods_scaled, non_mods_scaled)],
    ], right_align_columns=[0, 1, 2])
    print()


def print_events_stats():
    print('''# Terminology
**Intersections** are any of the following:
* **Overlap** - a key is partially overlapped by another (e.g. Shift down, A down, Shift up, A up - Shift and A overlap each other)
* **Wrap** - a key is completely wrapped by another (e.g. Shift down, A down, A up, Shift up - Shift wraps A)
''')
    print()

    print('All durations are in milliseconds.')

    key_durations = {}
    no_intersect_press_durations = {MOD_VALUE: [], NON_MOD_VALUE: []}
    intersect_press_durations = {MOD_VALUE: [], NON_MOD_VALUE: []}
    # The key of this is (first is mod or not, second is mod or not)
    intersections = {
        # Each of these dicts maps (first_pressed_key, other_key) to a list of tuples:
        #     (intersection_duration, intersection_percentage, duration_between_both_pressed)
        (NON_MOD_VALUE, NON_MOD_VALUE): {},
        (NON_MOD_VALUE, MOD_VALUE): {},
        (MOD_VALUE, NON_MOD_VALUE): {},
        (MOD_VALUE, MOD_VALUE): {},
    }
    down = {}
    duration_between_previous_release_and_this_press = {}
    last_release_key = None
    last_release_timestamp = None
    total_counts = Counter()
    zero_overlap_counts = Counter()
    overlap_counts = Counter()
    wrap_counts = Counter()  # e.g. Shift down, A down, A up, Shift up
    mods_simultaneous_counts = Counter()
    simultaneous_counts = Counter()
    key_counts = Counter()
    mods_down = set()
    wrapped_durations_press_and_next_press = []

    for i, (timestamp, is_pressed, key) in enumerate(events):
        if is_pressed:
            if key in down:
                raise ValueError(
                    f'event {i}: key {key} pressed at {timestamp} but already down - down: {down}')

            down[key] = timestamp
            total_counts[key in MOD_KEYS] += 1

            key_counts[key] += 1
            if key in MOD_KEYS:
                mods_down.add(key)
            elif (len(down) > 2 or 'shift' not in down) and not down.keys().isdisjoint(MOD_KEYS):
                # mods are pressed right now, but this is not one
                # ignoring mods only overlapping mods here (same for non-mods)
                simultaneous_counts[" ".join(down.keys())] += 1
        else:
            # Key is released
            when_down = down.get(key)
            if when_down is None:
                raise ValueError(
                    f'event {i}: key {key} released at {timestamp} but not pressed - down: {down}')

            del down[key]
            if key in mods_down:
                if len(mods_down) > 1:
                    mods_simultaneous_counts[" ".join(sorted(mods_down))] += 1
                mods_down.remove(key)

            released_at = timestamp
            dur = released_at - when_down

            if last_release_key is not None:
                between_dur = when_down - last_release_timestamp
                # negative value mean that the previous key was released after this one was pressed down
                if 0 <= between_dur < 1500:
                    duration_between_previous_release_and_this_press.setdefault(key, []).append(
                        between_dur)
            last_release_key = key
            last_release_timestamp = timestamp

            key_durations.setdefault(key, []).append(dur)
            is_mod = key in MOD_KEYS

            if not down and events[i - 1][KEY_I] == key and events[i - 1][IS_DOWN_I]:
                zero_overlap_counts[is_mod] += 1
                no_intersect_press_durations[is_mod].append(dur)

            if down:
                intersect_press_durations[is_mod].append(dur)

            for other_key, other_when_down in down.items():
                other_is_mod = other_key in MOD_KEYS

                if other_when_down < when_down:
                    # the other key was down before this one, and it isn't released yet,
                    # so it completely wraps this one (e.g. Shift down, A down, A up, Shift up)
                    intersection_duration = dur
                    intersect_key = other_key, key
                    intersect_is_down = other_is_mod, is_mod

                    wrap_counts[(other_is_mod, is_mod)] += 1
                    wrapped_durations_press_and_next_press.append(
                        when_down - other_when_down)
                else:
                    # the other key was down after this one (e.g. Shift down, A down, Shift up, A up)
                    # Since this key (Shift in the example) will already be up, when the other key (A)
                    # is up, it would not be registered, if we only consider if the other key is mod
                    intersection_duration = released_at - other_when_down
                    intersect_key = key, other_key
                    intersect_is_down = is_mod, other_is_mod

                    overlap_counts[(is_mod, other_is_mod)] += 1

                intersection_percentage = -1
                if dur == 0:
                    print(f'{key} from {when_down} to {released_at} was pressed for 0 ms')
                else:
                    # the duration the just-now-released key is held down with a not-yet-released key
                    # divided by the total duration this key was held down
                    intersection_percentage = 100 * (intersection_duration / dur)

                duration_between_both_pressed = abs(other_when_down - when_down)

                intersections[intersect_is_down].setdefault(intersect_key, []).append(
                    (intersection_duration, intersection_percentage, duration_between_both_pressed)
                )

    if down:
        print(f"Somehow pressed_keys is not empty: {down}")
    print()

    all_durations = []
    mod_durations = []
    non_mod_durations = []

    print_h1("Press durations")
    if PRINT_INDIVIDUAL:
        print_h2("Per key")
    for key, durations in key_durations.items():
        if PRINT_INDIVIDUAL:
            print_h3(f"Key '{key}'")
            print_stats(durations)
        all_durations.extend(durations)
        d = mod_durations if key in MOD_KEYS else non_mod_durations
        d.extend(durations)
    if PRINT_INDIVIDUAL:
        print_h2("All keys")
    print_stats(all_durations)

    print_h2("Non-mod keys")
    print_stats(non_mod_durations)

    print_h3("Non-intersecting non-mod keys")
    print_stats(no_intersect_press_durations[NON_MOD_VALUE])

    print_h3("Intersecting non-mod keys")
    print_stats(intersect_press_durations[NON_MOD_VALUE])

    print_h2("Mod keys")
    print_stats(mod_durations)

    print_h3("Non-intersecting mod keys")
    print_stats(no_intersect_press_durations[MOD_VALUE])

    print_h3("Intersecting mod keys (mod is first or second)")
    print_stats(intersect_press_durations[MOD_VALUE])

    all_between_durations = []
    mod_between_durations = []
    non_mod_between_durations = []

    print()
    print_h1("Time between previous release and this press (non-intersecting keys)")
    if PRINT_INDIVIDUAL:
        print_h2("Per key")

    for key, durations in duration_between_previous_release_and_this_press.items():
        if PRINT_INDIVIDUAL:
            print_h3(f"Key '{key}'")
            print_stats(durations)

        all_between_durations.extend(durations)
        d = mod_between_durations if key in MOD_KEYS else non_mod_between_durations
        d.extend(durations)

    if PRINT_INDIVIDUAL:
        print_h2("All keys")
    print_stats(all_between_durations)

    print_h2("Non-mod keys")
    print_stats(non_mod_between_durations)

    print_h2("Mod keys")
    print_stats(mod_between_durations)

    print()
    print_h1("Intersections")
    print_intersections_all_reports(intersections)
    print()
    print_h1("Duration between presses of wrapped keys")
    print_stats(wrapped_durations_press_and_next_press)
    print_h1(f'Counts per key (top {N_MOST_COMMON})')
    print_counter_as_table(key_counts, ["key", "count"], N_MOST_COMMON)
    print()
    print_h1(f'Counts of keys that are pressed together (top {N_MOST_COMMON})')
    print_counter_as_table(simultaneous_counts, ["key", "count"], N_MOST_COMMON)
    print()
    print_h2(f'Counts of mod keys that are pressed together (top {N_MOST_COMMON})')
    print_counter_as_table(mods_simultaneous_counts, ["key", "count"], N_MOST_COMMON)
    print()
    print_h1("Counts per category")
    print_table([
        ['type', 'count', '%'],
        None,
        *p_rows(total_counts),
    ])
    print()
    print()
    print_h1("Counts per intersection type")
    print_table([
        ['type', 'count', '%'],
        None,
        *p_rows(zero_overlap_counts, "without intersection"),
        [],
        *p_multi_rows(overlap_counts, "overlaps"),
        [],
        *p_multi_rows(wrap_counts, "wraps"),
    ])
    print()

    # we want to know the real ratio of mods overlapping anything to non-mods overlapping anything
    moa = count_any_first(overlap_counts, MOD_VALUE)
    nmoa = count_any_first(overlap_counts, NON_MOD_VALUE)

    # same for wrap
    mwa = count_any_first(wrap_counts, MOD_VALUE)
    nmwa = count_any_first(wrap_counts, NON_MOD_VALUE)

    print_counts_vs_with_ratio(total_counts, 'overlapping', moa, nmoa)
    print_counts_vs_with_ratio(total_counts, 'wrapping', mwa, nmwa)

    print_h1("Counts per overlap type for non-mod keys")
    print_table(p_variations(NON_MOD_VALUE, zero_overlap_counts, overlap_counts, wrap_counts))
    print()

    print_h2("Counts per overlap type for mod keys")
    print_table(p_variations(MOD_VALUE, zero_overlap_counts, overlap_counts, wrap_counts))
    print()


def print_train_stats():
    print_h1("Training data")
    print_table([
        ['type', 'count', '%'],
        None,
        ['total', training_data.count],
        [],
        pct2_row('mod', training_data.mod_count, training_data.non_mod_count),
        pct2_row('non-mod', training_data.non_mod_count, training_data.mod_count),
        [],
        pct2_row('mod overlap', training_data.mod_overlap_count,
                 training_data.non_mod_overlap_count),
        pct2_row('non-mod overlap', training_data.non_mod_overlap_count,
                 training_data.mod_overlap_count),
        [],
        pct2_row('mod wrap', training_data.mod_wrap_count, training_data.non_mod_wrap_count),
        pct2_row('non-mod wrap', training_data.non_mod_wrap_count, training_data.mod_wrap_count),
        [],
        pct2_row('mod triple-down', training_data.mod_triple_down_count,
                 training_data.non_mod_triple_down_count),
        pct2_row('non-mod triple-down', training_data.non_mod_triple_down_count,
                 training_data.mod_triple_down_count),
    ])
    print()

    print_h2("Most common mods")
    print_counter_as_table(mod_training_data_counter, ["key", "count"], 50)
    print()

    for title, values in TRAIN_STATS.items():
        print_h2(title)
        fmt = DURATION_NUM_FMT
        if 'Percent' in title:
            fmt = '.2f'
        print_stats(values, fmt=fmt)

    print()


if __name__ == '__main__':
    events = load_events(LOG_PATH)
    print(f"Loaded {len(events):_} events from {LOG_PATH.name}")
    print(f"Those are split into {len(event_after_this_one_was_removed) + 1:_} sessions.")
    print()
    if PRINT_STATISTICS:
        print_events_stats()

    ev_to_pp_ov = collect_pp_ov_training_data(events)
    mod_training_data_counter = Counter()
    training_data = collect_training_data_by_pattern(events, mod_training_data_counter, ev_to_pp_ov)

    print("---")
    print()

    if PRINT_STATISTICS:
        print_train_stats()

    write_csv_of_training_data(training_data)
