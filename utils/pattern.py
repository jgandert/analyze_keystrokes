from collections.abc import Generator
from itertools import product
from typing import Union, Optional, Iterable


class OneOf:
    def __init__(self, content):
        self.size = 1
        self.not_chosen_count = 0  # count
        self.chosen = None
        self.content = content
        self.cur = 0  # can be used to keep track of which one we're processing

    @property
    def has_chosen(self):
        return self.chosen is not None

    def reset(self):
        self.not_chosen_count = 0
        self.chosen = None
        self.cur = 0

    def __repr__(self):
        if self.has_chosen:
            c = f'chosen={self.chosen}'
        else:
            c = f'not_chosen_count={self.not_chosen_count}'
        return f'OneOf<{hex(id(self))}>({c}, size={self.size}, content={repr(self.content)})'

    def add_one(self):
        self.size += 1

    def deactivate(self):
        """
        Return False if all are False
        """
        if self.has_chosen:
            raise ValueError("someone was already chosen")

        self.not_chosen_count += 1
        return self.not_chosen_count < self.size

    def activate(self):
        if self.has_chosen:
            raise ValueError("someone was already chosen")

        # if we deactivated 1, self.not_chosen would be 1
        # so this is the index of the one we choose
        self.chosen = self.not_chosen_count


def list_with_duplicates_as_one_of(pattern: Union[str, list[str]]) -> list[Union[str, OneOf]]:
    result = []
    upper: dict[str, OneOf] = {}
    for pi, p in enumerate(pattern):
        if pattern.count(p) == 1:
            result.append(p)
        else:
            en = upper.get(p)
            if en is None:
                en = upper[p] = OneOf(p)
            else:
                # we're using the same object again if it already exists
                en.add_one()
            result.append(en)
    return result


def permutations_of_pattern(pattern: str) -> list[str]:
    """
    If an uppercase letter in a pattern appears more than 2 times, the permutations here can have duplicates.

    Consider 'fFpFPFhPnHNH'. For the following two cases (in brackets the one not chosen)

    Chosen 1, 1, 0 of each alternative -> f(F)pF(PF)hPnHN(H) = 'fpFhPnHN'
    Chosen 2, 1, 0 of each alternative -> f(F)p(FP)FhPnHN(H) = 'fpFhPnHN'

    Apart from those, fpFhPnNH will also appear twice for the same reason.
    """
    letter_or_one_of = list_with_duplicates_as_one_of(pattern)

    all_one_of = []
    for lo in letter_or_one_of:
        if isinstance(lo, OneOf) and lo not in all_one_of:
            all_one_of.append(lo)

    result = []
    for chosen in product(*[range(lo.size) for lo in all_one_of]):
        r = ''
        for lo, c in zip(all_one_of, chosen):
            lo.reset()
            lo.chosen = c

        for lo in letter_or_one_of:
            if isinstance(lo, OneOf):
                if lo.cur == lo.chosen:
                    r += lo.content
                lo.cur += 1
            else:
                r += lo
        result.append(r)
    return result


class PermutationPatternMatcher:
    def __init__(self, pattern: str, down_index: int = 0, key_index: int = 1,
                 released_letters_to_ignore: Iterable[str] = frozenset()):
        if not pattern:
            raise ValueError("pattern cannot be empty")

        self.key_index = key_index
        self.down_index = down_index
        self.pattern_length = len(pattern)
        self.match_length = len(set(pattern)) + len(released_letters_to_ignore)
        self.letter_or_one_of = list_with_duplicates_as_one_of(pattern)
        self.unique_one_of = []
        for lo in self.letter_or_one_of:
            if isinstance(lo, OneOf) and lo not in self.unique_one_of:
                self.unique_one_of.append(lo)
        self.released_letters_to_ignore = frozenset({
            rl.lower() for rl in released_letters_to_ignore
        })

    def get_match(self, events: list[tuple], start_index: int = 0) -> Optional[dict[str, int]]:
        """
        Checks if a sequence of events matches a given pattern starting from a specified index.

        The pattern string uses lowercase letters for key down events and uppercase
        letters for key up events. The lowercase and uppercase versions of the same
        letter refer to the same key. For example, 'a' and 'A' both refer to the
        same key, but 'a' represents a key down event (first element of the tuple
        is True), and 'A' represents a key up event (first element of the tuple is False).

        If consecutive uppercase letters appear in the pattern, it indicates that the
        corresponding up events can occur in any order. For example, in 'abBA', after
        'ab', the 'B' and 'A' up events can occur in either 'BA' or 'AB' order.

        For example, with pattern 'abBAB':
        - [(True, 'ctrl'), (True, 'e'), (False, 'e'), (False, 'ctrl')] matches.
        - [(True, 'ctrl'), (True, 'e'), (False, 'ctrl'), (False, 'e')] matches.
        - [(True, 'ctrl'), (True, 'e'), (False, 'j'), (False, 'ctrl'), (False, 'e')] does not match.

        Args:
            events: A list of tuples, where each tuple is (is_down, key_name).
            start_index: The index in the events list to start matching from.

        Returns:
            A mapping of the letter to events' indexes if the events from start_index match the pattern, None otherwise.
        """
        if self.match_length > len(events):
            return None

        released_letters_to_ignore = set(self.released_letters_to_ignore)
        for oo in self.unique_one_of:
            oo.reset()

        letter_to_index = {}
        letter_to_key = {}
        loi = 0
        for i in range(start_index, min(start_index + self.match_length, len(events))):
            event = events[i]

            if (not event[self.down_index] and
                    self.is_release_ignored(event, released_letters_to_ignore, letter_to_key)):
                continue

            while loi < self.pattern_length:
                lo = self.letter_or_one_of[loi]
                loi += 1

                if isinstance(lo, OneOf):
                    if lo.has_chosen:
                        continue
                    elif self._match(event, lo.content, letter_to_key):
                        letter_to_index[lo.content] = i
                        lo.activate()
                        break
                    elif not lo.deactivate():
                        # deactivate returns False, when all are False
                        # and when that is the case, the pattern cannot be a match
                        return None
                elif self._match(event, lo, letter_to_key):
                    letter_to_index[lo] = i
                    break
                else:
                    return None

            if loi >= self.pattern_length:
                break

        return letter_to_index

    def is_release_ignored(self, event, released_letters_to_ignore, letter_to_key):
        key = event[self.key_index]

        for rl in released_letters_to_ignore:
            pressed_key = letter_to_key.get(rl)
            if pressed_key is not None and pressed_key == key:
                released_letters_to_ignore.discard(rl)
                return True
        return False

    def _match(self, event, letter, letter_to_key):
        # down = lower = True, up = upper = False
        if event[self.down_index] != letter.islower():
            return False

        key = event[self.key_index]
        ll = letter.lower()

        if ll not in letter_to_key:
            letter_to_key[ll] = key
            return True

        return key == letter_to_key[ll]

    def all_matches(self,
                    events: list[tuple],
                    start_index: int = 0,
                    continue_after_offset: int = 0,
                    continue_at_letter: Optional[str] = None
                    ) -> Generator[dict[str, int]]:
        step_size = self.match_length + continue_after_offset
        i = start_index
        last_possible_i = len(events) - self.match_length - 1
        while i <= last_possible_i:
            m = self.get_match(events, start_index=i)
            if m is None:
                i += 1
            else:
                yield m
                if continue_at_letter is None:
                    i += step_size
                else:
                    i = m[continue_at_letter] + continue_after_offset


if __name__ == '__main__':
    print(permutations_of_pattern('fFpFPFxPnzXzNzXz'))

    perm = permutations_of_pattern('fFpFPFhPnHNH')
    print(perm)
    expected = ['fFpPhnHN', 'fFpPhnNH', 'fFphPnHN', 'fFphPnNH', 'fpFPhnHN', 'fpFPhnNH', 'fpFhPnHN',
                'fpFhPnNH',
                'fpPFhnHN', 'fpPFhnNH', 'fpFhPnHN', 'fpFhPnNH']
    if perm != expected:
        print(f"FAILED: {perm} vs {expected}")

    test_cases = [
        {  # events shorter than pattern
            'pattern': 'abBA',
            'events': [(True, 'ctrl'), (False, 'ctrl')],
            'start_index': 0,
            'expected': None
        },
        {  # wrapped
            'pattern': 'abBA',
            'events': [(True, 'ctrl'), (True, 'e'), (False, 'e'), (False, 'ctrl')],
            'start_index': 0,
            'expected': {'a': 0, 'b': 1, 'B': 2, 'A': 3}
        },
        {
            'pattern': 'abBA',
            'events': [(True, 'ctrl'), (True, 'e'), (False, 'ctrl'), (False, 'e')],
            'start_index': 0,
            'expected': None
        },
        {  # overlapped
            'pattern': 'abAB',
            'events': [(True, 'ctrl'), (True, 'e'), (False, 'ctrl'), (False, 'e')],
            'start_index': 0,
            'expected': {'a': 0, 'b': 1, 'A': 2, 'B': 3}
        },
        {
            'pattern': 'abBA',
            'events': [(True, 'ctrl'), (True, 'e'), (False, 'ctrl'), (False, 'e')],
            'start_index': 0,
            'expected': None
        },
        {  # the j is unexpected
            'pattern': 'abBA',
            'events': [(True, 'ctrl'), (True, 'e'), (False, 'j'), (False, 'ctrl'), (False, 'e')],
            'start_index': 0,
            'expected': None
        },
        {
            'pattern': 'fF',
            'events': [(True, 'f'), (False, 'f')],
            'start_index': 0,
            'expected': {'f': 0, 'F': 1}
        },
        {
            'pattern': 'fF',
            'events': [(True, 'f'), (True, 'g')],
            'start_index': 0,
            'expected': None
        },
        {
            'pattern': 'fF',
            'events': [(True, 'f'), (False, 'g')],
            'start_index': 0,
            'expected': None
        },
        {
            'pattern': 'fF',
            'events': [(True, 'f'), (False, 'f'), (True, 'g')],
            'start_index': 0,
            'expected': {'f': 0, 'F': 1}
        },
        {  # start with multiple key up
            'pattern': 'Fx',
            'events': [(False, 'f'), (True, 'g'), (False, 'g')],
            'start_index': 0,
            'expected': {'F': 0, 'x': 1}
        },
        {  # start with multiple key up and end with multiple key down
            'pattern': 'FxFzXz',
            'events': [(False, 'f'), (True, 'x'), (True, 'z'), (False, 'x')],
            'start_index': 0,
            'expected': {'F': 0, 'x': 1, 'z': 2, 'X': 3}
        },
        {
            'pattern': 'FxFzXz',
            'events': [(True, 'x'), (False, 'f'), (True, 'z'), (False, 'x')],
            'start_index': 0,
            'expected': {'x': 0, 'F': 1, 'z': 2, 'X': 3}
        },
        {
            'pattern': 'FxFzXz',
            'events': [(True, 'x'), (False, 'f'), (False, 'x'), (True, 'z')],
            'start_index': 0,
            'expected': {'x': 0, 'F': 1, 'X': 2, 'z': 3}
        },
        {
            'pattern': 'fF',
            'events': [(True, 'f')],
            'start_index': 0,
            'expected': None
        },
        {
            'pattern': 'pPx',
            'events': [(True, 'a'), (False, 'a'), (True, 'b'), (False, 'b'), (True, 'a')],
            'start_index': 2,
            'expected': {'p': 2, 'P': 3, 'x': 4}
        },
        {
            'pattern': 'xYX',
            'events': [(True, 'a'), (False, 'a'), (True, 'b'), (False, 'a'), (False, 'b')],
            'start_index': 2,
            'expected': {'x': 2, 'Y': 3, 'X': 4}
        },
        {
            'pattern': 'pP',
            'events': [(True, 'a'), (False, 'a'), (True, 'c'), (False, 'b'), (True, 'a')],
            'start_index': 2,
            'expected': None
        },
        {  # h could be in any of 3 positions: first - X release is not part of the pattern
            'pattern': 'hnHNHxH',
            'events': [(True, 'h'), (True, 'n'), (False, 'h'), (False, 'n'), (True, 'x'),
                       (False, 'x')],
            'start_index': 0,
            'expected': {'h': 0, 'n': 1, 'H': 2, 'N': 3, 'x': 4}
        },
        {  # h could be in any of 3 positions: second
            'pattern': 'hnHNHxH',
            'events': [(True, 'h'), (True, 'n'), (False, 'n'), (False, 'h'), (True, 'x'),
                       (False, 'x')],
            'start_index': 0,
            'expected': {'h': 0, 'n': 1, 'N': 2, 'H': 3, 'x': 4}
        },
        {  # h could be in any of 3 positions: third
            'pattern': 'hnHNHxH',
            'events': [(True, 'h'), (True, 'n'), (False, 'n'), (True, 'x'), (False, 'h'),
                       (False, 'x')],
            'start_index': 0,
            'expected': {'h': 0, 'n': 1, 'N': 2, 'x': 3, 'H': 4}
        },
        {  # different letters can match the same key
            'pattern': 'abc',
            'events': [(True, 'x'), (True, 'a'), (True, 'x')],
            'start_index': 0,
            'expected': {'a': 0, 'b': 1, 'c': 2}
        },
        {
            'pattern': 'fFpFPFhPnHNH',
            'events': [(True, 'f'), (False, 'f'), (True, 'p'), (False, 'p'), (True, 'h'),
                       (True, 'n'), (False, 'h'), (True, 'n')],
            # the last n should be an up event (False)
            'start_index': 0,
            'expected': None
        },
        {  # actual: fFpPhnHN
            'pattern': 'fFpFPFhPnHNH',
            'events': [(True, 'f'), (False, 'f'), (True, 'p'), (False, 'p'), (True, 'h'),
                       (True, 'n'), (False, 'h'),
                       (False, 'n')],
            'start_index': 0,
            'expected': {'f': 0, 'F': 1, 'p': 2, 'P': 3, 'h': 4, 'n': 5, 'H': 6, 'N': 7}
        },
        {  # actual: fFpPhnNH
            'pattern': 'fFpFPFhPnHNH',
            'events': [(True, 'f'), (False, 'f'), (True, 'p'), (False, 'p'), (True, 'h'),
                       (True, 'n'), (False, 'n'),
                       (False, 'h')],
            'start_index': 0,
            'expected': {'f': 0, 'F': 1, 'p': 2, 'P': 3, 'h': 4, 'n': 5, 'N': 6, 'H': 7}
        },
        {  # actual: fFphPnHN
            'pattern': 'fFpFPFhPnHNH',
            'events': [(True, 'f'), (False, 'f'), (True, 'p'), (True, 'h'), (False, 'p'),
                       (True, 'n'), (False, 'h'),
                       (False, 'n')],
            'start_index': 0,
            'expected': {'f': 0, 'F': 1, 'p': 2, 'h': 3, 'P': 4, 'n': 5, 'H': 6, 'N': 7}
        },
        {  # actual: fFphPnNH
            'pattern': 'fFpFPFhPnHNH',
            'events': [(True, 'f'), (False, 'f'), (True, 'p'), (True, 'h'), (False, 'p'),
                       (True, 'n'), (False, 'n'),
                       (False, 'h')],
            'start_index': 0,
            'expected': {'f': 0, 'F': 1, 'p': 2, 'h': 3, 'P': 4, 'n': 5, 'N': 6, 'H': 7}
        },
        {  # actual: fpFPhnHN
            'pattern': 'fFpFPFhPnHNH',
            'events': [(True, 'f'), (True, 'p'), (False, 'f'), (False, 'p'), (True, 'h'),
                       (True, 'n'), (False, 'h'),
                       (False, 'n')],
            'start_index': 0,
            'expected': {'f': 0, 'p': 1, 'F': 2, 'P': 3, 'h': 4, 'n': 5, 'H': 6, 'N': 7}
        },
        {  # actual: fpFPhnNH
            'pattern': 'fFpFPFhPnHNH',
            'events': [(True, 'f'), (True, 'p'), (False, 'f'), (False, 'p'), (True, 'h'),
                       (True, 'n'), (False, 'n'),
                       (False, 'h')],
            'start_index': 0,
            'expected': {'f': 0, 'p': 1, 'F': 2, 'P': 3, 'h': 4, 'n': 5, 'N': 6, 'H': 7}
        },
        {  # actual: fpFhPnHN
            'pattern': 'fFpFPFhPnHNH',
            'events': [(True, 'f'), (True, 'p'), (False, 'f'), (True, 'h'), (False, 'p'),
                       (True, 'n'), (False, 'h'),
                       (False, 'n')],
            'start_index': 0,
            'expected': {'f': 0, 'p': 1, 'F': 2, 'h': 3, 'P': 4, 'n': 5, 'H': 6, 'N': 7}
        },
        {  # actual: fpFhPnNH
            'pattern': 'fFpFPFhPnHNH',
            'events': [(True, 'f'), (True, 'p'), (False, 'f'), (True, 'h'), (False, 'p'),
                       (True, 'n'), (False, 'n'),
                       (False, 'h')],
            'start_index': 0,
            'expected': {'f': 0, 'p': 1, 'F': 2, 'h': 3, 'P': 4, 'n': 5, 'N': 6, 'H': 7}
        },
        {  # actual: fpPFhnHN
            'pattern': 'fFpFPFhPnHNH',
            'events': [(True, 'f'), (True, 'p'), (False, 'p'), (False, 'f'), (True, 'h'),
                       (True, 'n'), (False, 'h'),
                       (False, 'n')],
            'start_index': 0,
            'expected': {'f': 0, 'p': 1, 'P': 2, 'F': 3, 'h': 4, 'n': 5, 'H': 6, 'N': 7}
        },
        {  # actual: fpPFhnNH
            'pattern': 'fFpFPFhPnHNH',
            'events': [(True, 'f'), (True, 'p'), (False, 'p'), (False, 'f'), (True, 'h'),
                       (True, 'n'), (False, 'n'),
                       (False, 'h')],
            'start_index': 0,
            'expected': {'f': 0, 'p': 1, 'P': 2, 'F': 3, 'h': 4, 'n': 5, 'N': 6, 'H': 7}
        },
        {  # pattern with multiple lowercase letters
            'pattern': 'azAz',
            'events': [(True, 'a'), (True, 'z'), (False, 'a')],
            'start_index': 0,
            'expected': {'a': 0, 'z': 1, 'A': 2}
        },
        {
            'pattern': 'azAz',
            'events': [(True, 'a'), (False, 'a'), (True, 'z')],
            'start_index': 0,
            'expected': {'a': 0, 'A': 1, 'z': 2}
        },
        {  # this one previously matched, so parts of the pattern where not found (which was a bug)
            'pattern': 'pPxPnN',  # p=l, x=l
            'events': [(True, 'l'), (False, 'l'), (True, 'l'), (False, 'l'), (True, 'n'),
                       (False, 'n')],
            'start_index': 0,
            'expected': None
        },
        {  # same one appears multiple times
            'pattern': 'pPxX',
            'events': [(True, 'l'), (False, 'l'), (True, 'l'), (False, 'l'), (True, 'n'),
                       (False, 'n')],
            'start_index': 0,
            'expected': {'p': 0, 'P': 1, 'x': 2, 'X': 3}
        },
        {
            'pattern': '',
            'events': [(True, 'l'), (False, 'l')],
            'start_index': 0,
            'expected': 'pattern cannot be empty'
        },
        {
            # this looks like a bug because the result is {'x': 0, 'y': 1} not {'y': 0, 'x': 1} but
            # that's just because we used keys that look like they should match the letters of the
            # pattern, but there's no reason why they should. Letters in a pattern are like
            # variables, and here letter 'x' happened to match the 'y' of the event.
            'pattern': 'xyx',
            'events': [(True, 'y'), (True, 'x')],
            'start_index': 0,
            'expected': {'x': 0, 'y': 1},
        },
        {  # no X in the pattern, but should be ignored
            'pattern': 'pPxt',
            'events': [(True, 'p'), (False, 'p'), (True, 'x'), (False, 'x'), (True, 't')],
            'start_index': 0,
            'expected': {'p': 0, 'P': 1, 'x': 2, 't': 4},
            'released_letters_to_ignore': frozenset(['x']),
        },
        {  # same here
            'pattern': 'pPxt',
            'events': [(True, 'p'), (False, 'p'), (True, 'x'), (True, 't'), (False, 'x')],
            'start_index': 0,
            'expected': {'p': 0, 'P': 1, 'x': 2, 't': 3},
            'released_letters_to_ignore': frozenset(['x']),
        },
        {  # even without ignoring x releases
            'pattern': 'pPxt',
            'events': [(True, 'p'), (False, 'p'), (True, 'x'), (True, 't'), (False, 'x')],
            'start_index': 0,
            'expected': {'p': 0, 'P': 1, 'x': 2, 't': 3},
        },
        {  # no X and Y in the pattern, but should be ignored
            'pattern': 'xyx',
            'events': [(True, 'x'), (False, 'x'), (True, 'y'), (False, 'y')],
            'start_index': 0,
            'expected': {'x': 0, 'y': 2},
            'released_letters_to_ignore': frozenset(['x', 'y']),
        },
        {  # same here
            'pattern': 'xyx',
            'events': [(True, 'x'), (True, 'y'), (False, 'x'), (False, 'y')],
            'start_index': 0,
            'expected': {'x': 0, 'y': 1},
            'released_letters_to_ignore': frozenset(['x', 'y']),
        },
        {  # same here
            'pattern': 'yxy',
            'events': [(True, 'y'), (True, 'x'), (False, 'x'), (False, 'y')],
            'start_index': 0,
            'expected': {'y': 0, 'x': 1},
            'released_letters_to_ignore': frozenset(['x', 'y']),
        },
        {  # a bit more complex pattern
            'pattern': 'pxPyPx',
            'events': [(True, 'p'), (True, 'x'), (False, 'p'), (False, 'x'), (True, 'y'),
                       (False, 'y')],
            'start_index': 0,
            'expected': {'p': 0, 'x': 1, 'P': 2, 'y': 4},
            'released_letters_to_ignore': frozenset(['x', 'y']),
        },
        {  # again with different order
            'pattern': 'pxPyPx',
            'events': [(True, 'p'), (True, 'x'), (False, 'x'), (True, 'y'), (False, 'p'),
                       (False, 'y')],
            'start_index': 0,
            'expected': {'p': 0, 'x': 1, 'y': 3, 'P': 4},
            'released_letters_to_ignore': frozenset(['x', 'y']),
        },
    ]

    for i, case in enumerate(test_cases):
        result = ''
        try:
            released_letters_to_ignore = case.get('released_letters_to_ignore', frozenset())
            result = PermutationPatternMatcher(
                case['pattern'],
                released_letters_to_ignore=released_letters_to_ignore
            ).get_match(case['events'], case['start_index'])
            status = "PASSED" if result == case['expected'] else "FAILED"
        except ValueError as e:
            status = "PASSED" if str(e).startswith(str(case['expected'])) else f"FAILED WITH {e}"

        if status == "PASSED":
            continue

        print(
            f"Test {i + 1}: Pattern='{case['pattern']}', Events={case['events']}, Start={case['start_index']}, Expected={case['expected']}, Result={result}, {status}")
