from enum import Enum


class LossType(Enum):
    BCE = "BCE"
    SoftBEROnAllZero = "SoftBEROnAllZero"
    FEROnAllZero = "FEROnAllZero"
