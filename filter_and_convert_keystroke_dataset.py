import csv
import json
from pathlib import Path

from utils.constants import SUPPORTED_LAYOUTS, VK_CODE_TO_LETTER, LAYOUT_TO_UPPERCASE
from utils.numbers import timestamp_of, to_ms
from utils.serialization import TabMinDialect, write_to_csv_gz, fix_csv_field_size

DATASET_PATH = Path(__file__).parent / 'dataset'

EXTRACTED_ARCHIVE_PATH = DATASET_PATH / 'extract_archive_in_here'
CSV_DIR_PATH = EXTRACTED_ARCHIVE_PATH / "files"
PARTICIPANTS_CSV_PATH = EXTRACTED_ARCHIVE_PATH / "metadata_participants.txt"

RESULT_PATH = DATASET_PATH / "filtered_events.csv.gz"
RESULT_USED_PARTICIPANTS_PATH = DATASET_PATH / "filtered_events.participants_used.json"
RESULT_USED_PARTICIPANTS_METADATA = DATASET_PATH / "filtered_events.participants_used_metadata.csv"

MAP_LETTER = {'ARW_LEFT': 'left', 'ARW_RIGHT': 'right', 'ARW_UP': 'up', 'ARw_DOWN': 'down',
              ' ': 'space', 'BKSP': 'backspace'}

KS_HEADER = ['PARTICIPANT_ID', 'TEST_SECTION_ID', 'SENTENCE', 'USER_INPUT', 'KEYSTROKE_ID',
             'PRESS_TIME',
             'RELEASE_TIME', 'LETTER', 'KEYCODE']

PARTICIPANT_IDX = KS_HEADER.index('PARTICIPANT_ID')
SECTION_IDX = KS_HEADER.index('TEST_SECTION_ID')
PRESS_TIME_IDX = KS_HEADER.index('PRESS_TIME')
RELEASE_TIME_IDX = KS_HEADER.index('RELEASE_TIME')
LETTER_IDX = KS_HEADER.index('LETTER')
KEYCODE_IDX = KS_HEADER.index('KEYCODE')

KS_HEADER_TO_INDEX = dict(enumerate(KS_HEADER))

fix_csv_field_size()

num_participants_kept = 0
valid_participant_file_names = set()
participant_id_to_layout = {}

num_original_participants = 0
num_down = 0

participant_id_to_metadata = {}

with PARTICIPANTS_CSV_PATH.open(mode='r', newline='', encoding='utf-8') as file:
    reader = csv.DictReader(file, delimiter='\t')

    for row in reader:
        num_original_participants += 1

        if not row['FINGERS'] or not row['ERROR_RATE'] or not row['AVG_WPM_15']:
            continue

        layout = row['LAYOUT']
        if layout not in SUPPORTED_LAYOUTS:
            continue
        layout = layout.strip()

        if row['KEYBOARD_TYPE'] == 'on-screen':
            continue

        fingers = row['FINGERS'].strip()
        error_rate = float(row['ERROR_RATE'])
        avg_wpm_15 = float(row['AVG_WPM_15'])

        if avg_wpm_15 > 114 and error_rate == 0:
            # may have cheated (also not representable for most typists)
            continue

        # Average WPM is around 40. In the dataset it is 60.25 (median 59.89)
        # According to the study: Overlapping key presses indicate faster typing.
        # We want a lot of overlap, and we want faster typists.
        if fingers != "1-2" and error_rate < 5 and avg_wpm_15 > 35:
            num_participants_kept += 1
            pid = row['PARTICIPANT_ID']
            valid_participant_file_names.add(f'{pid}_keystrokes.txt')
            participant_id_to_layout[pid] = layout
            participant_id_to_metadata[pid] = dict(row)

print(num_participants_kept, 'of', num_original_participants)

keystrokes_result = []

max_dur_participant = None
max_dur_participant_dur = None

cur_t = 0
used_participants = set()


def parse_file(csv_path, writer):
    global cur_t, num_down

    if csv_path.name not in valid_participant_file_names:
        return

    participant_id = get_participant_id(csv_path)
    layout = participant_id_to_layout.get(participant_id)
    if not layout:
        return

    with csv_path.open(mode='r', newline='', encoding='ansi') as file:
        reader = csv.reader(file, delimiter='\t', quoting=csv.QUOTE_NONE)

        # Skip the header row
        header = next(reader)
        if header != KS_HEADER:
            print_file_ignored(csv_path, f"Header is not correct: {header}")
            return

        section_id_to_rows = get_sections(csv_path, reader)

    if not section_id_to_rows:
        return

    # Sort the rows of each section
    events = sum(section_id_to_rows.values(), start=[])
    events.sort(key=lambda ev: ev[0])

    check_result = check_for_issues(events, layout)
    if check_result is not None:
        print_file_ignored(csv_path, check_result)
        return

    first_t = events[0][0]
    last_t = events[-1][0]

    if first_t < timestamp_of(2000, 1, 1):
        print_file_ignored(csv_path, 'Made before year 2000')
        return

    cur_delta = cur_t - first_t

    write_rows(writer, events, cur_delta)
    num_down += len(events) // 2

    # mark the end of this participant's block
    writer.writerow((last_t + cur_delta + to_ms(seconds=30), None, None))

    used_participants.add(participant_id)
    cur_t = last_t + cur_delta + to_ms(minutes=30)


def get_participant_id(path: Path) -> str:
    return path.name.split('_')[0]


def get_sections(csv_path, reader):
    section_id_to_rows = {}
    ignored_section_ids = set()
    num_presses = 0

    num_low_delta = 0

    check_if_section_was_finished = False
    prev_section_id = None
    participant_id = get_participant_id(csv_path)

    for row in reader:
        if len(row) < 8 or row[-1] == '':
            check_if_section_was_finished = True
            continue

        if row[PARTICIPANT_IDX] != participant_id:
            print_file_ignored(csv_path,
                               f'Participant ID differs from filename: {row[PARTICIPANT_IDX]}')
            return None

        section_id = row[SECTION_IDX]
        if not section_id or section_id in ignored_section_ids:
            continue

        if check_if_section_was_finished:
            check_if_section_was_finished = False
            if section_id == prev_section_id:
                print_section_ignored(csv_path, section_id, 'There was an incomplete row')
                del section_id_to_rows[section_id]
                ignored_section_ids.add(section_id)
                continue

        section = section_id_to_rows.get(section_id)
        if section is None:
            section = []
            section_id_to_rows[section_id] = section

        # first use float as int("1.473278e+12") will fail
        press_time_ms = int(float(row[PRESS_TIME_IDX]))
        release_time_ms = int(float(row[RELEASE_TIME_IDX]))

        delta = release_time_ms - press_time_ms

        letter = row[LETTER_IDX]

        if delta < 0:
            # release before press
            del section_id_to_rows[section_id]
            ignored_section_ids.add(section_id)
            continue
        elif delta > to_ms(seconds=7):
            # We're assuming that given that this happened,
            # all times of this participant are unreliable
            print_file_ignored(csv_path,
                               f'{letter} at {press_time_ms} took way too long: {delta} ms')
            return None
        elif delta < 1:
            print_section_ignored(csv_path, section_id,
                                  f'{letter} at {press_time_ms} was way too short: {delta} ms')
            del section_id_to_rows[section_id]
            ignored_section_ids.add(section_id)
            continue

        if letter == '' or not letter.isprintable():
            # Sometimes there seem to be key presses missing (see participant 100032)
            # but this may be a general problem (or due to using the mouse to move the cursor)

            # print_ignored(csv_path, 'No letter, only key codes')
            # return

            code = row[KEYCODE_IDX]
            letter = VK_CODE_TO_LETTER.get(code)

        if letter is None:
            del section_id_to_rows[section_id]
            ignored_section_ids.add(section_id)
            continue

        if delta < 10:
            num_low_delta += 1

        prev_section_id = section_id
        num_presses += 1
        section.append([press_time_ms, 1, letter])
        section.append([release_time_ms, 0, letter])

    if not section_id_to_rows or not num_presses:
        return None

    if (num_low_delta / num_presses) > 0.4:
        print_file_ignored(csv_path, 'A large percentage of key presses were too short')
        return None

    return section_id_to_rows


def print_file_ignored(csv_path, message):
    print(f'{csv_path.name} IGNORED: {message}')


def print_section_ignored(csv_path, section_id, message):
    print(f'{csv_path.name} SECTION {section_id} IGNORED: {message}')


def check_for_issues(events, layout):
    down_lower_letters = {}
    num_overlap = 0
    uppercase = LAYOUT_TO_UPPERCASE[layout]

    for row in events:
        t, down, letter = row
        lower_letter = letter.lower()

        if down:
            if down_lower_letters:
                for k in down_lower_letters:
                    down_lower_letters[k] += 1
                num_overlap += 1

            if lower_letter in down_lower_letters:
                return f'{letter} down at {t} but was already'
            down_lower_letters[lower_letter] = 0

            if letter in uppercase and 'shift' not in down_lower_letters:
                return f'{letter} is uppercase at {t} but shift is not down'
        else:
            if lower_letter not in down_lower_letters:
                return f'{letter} up at {t} but is not even down'

            down_with_n_keys = down_lower_letters.pop(lower_letter)
            if lower_letter not in ('shift', 'ctrl') and down_with_n_keys > 3:
                return f'{letter} up at {t} was down with {down_with_n_keys} other keys at the same time'

    if down_lower_letters:
        return f'Some keys were never released: {down_lower_letters}'

    if num_overlap < 7:
        # This could be contested. It does skew the statistics (zero overlap vs overlap etc.),
        # but for our use case (tap-hold), we do not need zero overlaps.
        return f'Too few overlapping keys: {num_overlap}'

    return None


def write_rows(writer, events, delta):
    # We will not map keys here to their non-shifted version, as that leads to conflicts (keys pressed down twice)
    for row in events:
        dt, down, letter = row
        letter = MAP_LETTER.get(letter, letter)
        last_t = dt + delta

        try:
            writer.writerow((last_t, down, letter.lower().strip().replace(' ', '_')))
        except csv.Error as e:
            raise e


def size_in_mib(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def name_and_size(path: Path) -> str:
    return f'{path.name}: {size_in_mib(path)} MiB'


def process_writer(writer):
    for path in sorted(CSV_DIR_PATH.glob("*.txt"), key=lambda i: int(get_participant_id(i))):
        try:
            parse_file(path, writer)
        except csv.Error as e:
            print(f'{path.name} IGNORED: {e}')
        except Exception as e:
            print(path, e)
            raise e


write_to_csv_gz(RESULT_PATH, process_writer)

print(f"{len(used_participants)} participants left")
print(f'{num_down} keys down (and up)')
print(name_and_size(RESULT_PATH))

RESULT_USED_PARTICIPANTS_PATH.write_text(
    json.dumps(sorted(used_participants, key=int))
)

with RESULT_USED_PARTICIPANTS_METADATA.open('w', encoding='utf-8', newline='') as csvfile:
    fieldnames = ['PARTICIPANT_ID', 'AGE', 'GENDER', 'HAS_TAKEN_TYPING_COURSE', 'COUNTRY', 'LAYOUT',
                  'NATIVE_LANGUAGE',
                  'FINGERS', 'TIME_SPENT_TYPING', 'KEYBOARD_TYPE', 'ERROR_RATE', 'AVG_WPM_15',
                  'AVG_IKI', 'ECPC',
                  'KSPC', 'ROR']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, dialect=TabMinDialect)

    writer.writeheader()
    for pid, md in sorted(participant_id_to_metadata.items(),
                          key=lambda x: float(x[1]['AVG_WPM_15']), reverse=True):
        if pid in used_participants:
            writer.writerow(md)
