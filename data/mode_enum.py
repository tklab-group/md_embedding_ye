from enum import Enum


# mode normal sub-word sub-word-no-full(sub-word without fullname) n-gram
class Mode(Enum):
    NORMAL = 0
    SUB_WORD = 1
    SUB_WORD_NO_FULL = 2
    N_GRAM = 3
