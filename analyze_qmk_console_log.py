import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from utils.format import print_table, print_h1


@dataclass
class SectionInfo:
    has_second: bool
    choose_hold: bool
    has_third: bool
    second_opposite: bool
    decision_took_ms: Optional[str]
    pth_key: Optional[str]
    pth_side: Optional[str]


DIR_PATH = Path.home()
OUTPUT_PATH = DIR_PATH / "qmk.merged.log"

HEX_OFFSET_RE = re.compile(r'^[0-9A-Fa-f]{2}:')
PTH_KEY = re.compile(r"Key (.*?) is DOWN \(side=(.*?)\)")
DECISION_RE = re.compile(r"-> DECIDED_.*? after (\d+) ms")
COMMON_PREFIX = ": PTH: "


def filter_log(lines):
    for line in lines:
        idx = line.find(COMMON_PREFIX)
        if idx == -1:
            continue
        line = line[idx + len(COMMON_PREFIX):].rstrip('\n')
        if line.startswith('Key '):
            line = '\n' + line
        yield line + '\n'


def merge_and_clean_logs():
    with OUTPUT_PATH.open('w+', encoding='utf-8') as fout:
        for in_path in sorted(DIR_PATH.glob("qmk*.log")):
            if in_path.name == OUTPUT_PATH.name:
                continue

            with open(in_path, 'r', encoding='utf-8') as fin:
                for line in filter_log(fin):
                    fout.write(line)


def a(x):
    if not x:
        return ()
    # return f'median: {statistics.median(x)}, avg: {statistics.mean(x)}, min: {min(x)}, max: {max(x)}'
    return min(x), max(x), int(statistics.mean(x)), int(statistics.median(x))


def print_counts(result: list[SectionInfo]):
    # is hold, key, dur, has_second
    all_with_second = [i.decision_took_ms for i in result if i.has_second]
    all_without_second = [i.decision_took_ms for i in result if not i.has_second]
    all_ = [i.decision_took_ms for i in result]

    ## all with second which are hold TODO
    print_table([
        ["type", "min", "max", "avg", "median"],
        ["all_with_second     ", *a(all_with_second)],
        ["all_without_second  ", *a(all_without_second)],
        ["all                 ", *a(all_)],
    ])


if __name__ == '__main__':
    # paste your log here if you just care about getting the sanitized output
    lines = '''

'''.strip().splitlines()

    if len(lines) > 1:
        print(''.join(filter_log(lines)))
        print()
        print()
        print()
    merge_and_clean_logs()

    # now read and analyze them
    c = OUTPUT_PATH.read_text(encoding='utf-8').split(
        "--------------------------------------------------------------------------------")

    result = []
    for section in c:
        has_second = 'SECOND_PRESSED' in section
        choose_hold = 'DECIDED_HOLD' in section
        has_third = 'Third key pressed' in section
        second_opposite = 'Second is opposite-side press' in section

        dec = DECISION_RE.search(section)
        if dec is None:
            continue
        decision_took_ms = int(dec.group(1))

        pth_match = PTH_KEY.search(section)
        pth_key = pth_match.group(1)
        pth_side = pth_match.group(2)

        info = SectionInfo(
            has_second=has_second,
            choose_hold=choose_hold,
            has_third=has_third,
            second_opposite=second_opposite,
            decision_took_ms=decision_took_ms,
            pth_key=pth_key,
            pth_side=pth_side
        )
        result.append(info)

    tap_chosen = [i for i in result if not i.choose_hold]
    count_tap = len(tap_chosen)

    hold_chosen = [i for i in result if i.choose_hold]
    count_hold = len(hold_chosen)

    print_table([
        ['type', 'count'],
        ['hold', count_hold],
        ['tap', count_tap],
    ])
    print()

    print_counts(result)
    print()

    print_h1("Tap only")
    print_counts(tap_chosen)
    print()

    print_h1("Hold only")
    print_counts(hold_chosen)
