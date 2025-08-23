from enum import Enum, auto


class TrainingDataColId(Enum):
    is_mod = 0

    second_is_mod = auto()

    # These are about the press that was right before the PTH.
    # It may be released after the PTH press time.
    # If not, these will equal key_released_before_pth values.
    key_pressed_before_pth_is_mod = auto()
    key_pressed_before_pth_release_time = auto()

    # This is always about the release that happened before the PTH.
    key_released_before_pth_is_mod = auto()
    key_released_before_pth_release_time = auto()

    pth_press_time = auto()
    second_press_time = auto()
    second_release_time = auto()
    pth_release_time = auto()
    third_press_time = auto()

    # if you want ALL the training data, uncomment these and search for the
    # string "if you want ALL the training data"
    # pp1 = auto()
    # pp2 = auto()
    pp3 = auto()
    pp4 = auto()
    # ov1 = auto()
    # ov2 = auto()
    ov3 = auto()
    ov4 = auto()
    down_count = auto()

    # non_mod_dur_avg = auto()
    # mod_dur_avg = auto()
    # overlap_dur_avg = auto()

    # recent_dur_avg = auto()
    # recent_is_mod_avg = auto()
    # pressed_count_last_400_ms = auto()
    # pth_and_second_same_side_but_third_not = auto()

    @classmethod
    @property
    def member_names(cls) -> list[str]:
        return cls._member_names_

    def is_bool(self):
        return self in BOOLEAN_COLS

    def parse(self, val: str):
        if self.is_bool():
            return val == 'True'
        return int(val)

    def to_csv(self, val):
        if self.is_bool():
            return 1 if val else 0
        return val


BOOLEAN_COLS = {
    TrainingDataColId.is_mod, TrainingDataColId.second_is_mod,
    TrainingDataColId.key_released_before_pth_is_mod,
    TrainingDataColId.key_pressed_before_pth_is_mod
    # if you want ALL the training data, uncomment this
    # TrainingDataColId.pth_and_second_same_side_but_third_not
}
