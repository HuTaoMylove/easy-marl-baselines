class AC_buffer:
    def __init__(self):
        self.reset()

    def need_nextaction(self):
        return False

    def reset(self):
        self.transition_dict_1 = {
            'states': [],
            'actions': [],
            'next_states': [],
            'rewards': [],
            'dones': [],
            'next_action': []
        }
        self.transition_dict_2 = {
            'states': [],
            'actions': [],
            'next_states': [],
            'rewards': [],
            'dones': [],
            'next_action': []
        }

    def insert(self, data0: dict, data1: dict):
        for k, v in data0.items():
            self.transition_dict_1[k].append(v)
        for k, v in data1.items():
            self.transition_dict_2[k].append(v)

    def compute(self):
        assert 'asdf'

    def get_train_data(self):
        return [self.transition_dict_1, self.transition_dict_2]
