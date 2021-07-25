import enum

class Control_Phase(enum.Enum):
    """the locomotion state of finger"""
    POSITION = enum.auto()
    TORQUE = enum.auto()
    # TODO: create more situations support
    EARLY_CONTACT = enum.auto()
    LOSE_CONTACT = enum.auto()