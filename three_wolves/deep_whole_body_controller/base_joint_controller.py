import enum

class Control_Phase(enum.Enum):
    """the locomotion state of finger"""
    POSITION = 0
    TORQUE = 1
    # TODO: create more situations support
    EARLY_CONTACT = 2
    LOSE_CONTACT = 3

class BaseJointController(object):
    # def __init__(self):
    #     self.obs_dict = None

    def reset(self):
        raise NotImplemented()

    def update(self):
        raise NotImplemented()

    def get_action(self, policy_action):
        raise NotImplemented()