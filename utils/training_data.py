from dataclasses import dataclass


@dataclass
class TrainingData:
    elements: list
    mod_count: int = 0
    non_mod_count: int = 0
    mod_wrap_count: int = 0  # 567537
    non_mod_wrap_count: int = 0  # 119237
    mod_overlap_count: int = 0  # 1230364
    non_mod_overlap_count: int = 0  # 12282948
    mod_triple_down_count: int = 0
    non_mod_triple_down_count: int = 0

    @property
    def count(self):
        return len(self.elements)

    def add_count(self, is_mod, is_wrapped, is_triple_down):
        if is_mod:
            self.mod_count += 1
            if is_wrapped:
                self.mod_wrap_count += 1
            else:
                self.mod_overlap_count += 1

            if is_triple_down:
                self.mod_triple_down_count += 1
        else:
            self.non_mod_count += 1
            if is_wrapped:
                self.non_mod_wrap_count += 1
            else:
                self.non_mod_overlap_count += 1

            if is_triple_down:
                self.non_mod_triple_down_count += 1
