import csv
import gzip
import sys
from csv import Dialect
from pathlib import Path


def fix_csv_field_size():
    maxInt = sys.maxsize

    # find acceptable value for field size limit
    while True:
        try:
            csv.field_size_limit(maxInt)
            break
        except OverflowError:
            maxInt = int(maxInt / 10)


class TabMinDialect(Dialect):
    delimiter = '\t'
    lineterminator = '\n'
    quoting = csv.QUOTE_NONE
    quotechar = None


def deser_down(d):
    if d == "":
        return None
    return 1 if int(d) else 0


def deser_key(key):
    if key == "":
        return None
    return key


def read_keylog_from_csv_gz(path: Path):
    def r(reader):
        return [(int(t), deser_down(d), deser_key(k)) for t, d, k in reader]

    return read_from_csv_gz(path, r)


def read_from_csv_gz(path: Path, process_reader):
    with gzip.open(str(path), 'rt', newline='', encoding='utf-8') as file:
        reader = csv.reader(file, dialect=TabMinDialect)
        return process_reader(reader)


def write_items_to_csv_gz(path: Path, items):
    def wr(writer):
        for item in items:
            writer.writerow(item)

    write_to_csv_gz(path, wr)


def write_to_csv_gz(path: Path, process_writer):
    with gzip.open(str(path), 'wt', newline='', encoding='utf-8') as gz_file:
        writer = csv.writer(gz_file, dialect=TabMinDialect)
        process_writer(writer)
