class DictWrapper:
    def __init__(self, initial_dict=None):
        if initial_dict is None:
            initial_dict = {}
        self.__dict__.update(initial_dict)

    def __getattr__(self, item):
        try:
            return self.__dict__[item]
        except KeyError:
            raise AttributeError(f"'DictWrapper' object has no attribute '{item}'")

    def __setattr__(self, key, value):
        self.__dict__[key] = value

    def __delattr__(self, item):
        try:
            del self.__dict__[item]
        except KeyError:
            raise AttributeError(f"'DictWrapper' object has no attribute '{item}'")

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __delitem__(self, key):
        del self.__dict__[key]

    def to_dict(self):
        return self.__dict__


def ragas_dataset(questions, contexts, ground_truths=None):
    dataset = {'question': questions, 'context': contexts}

    if ground_truths is not None:
        dataset['ground_truth'] = ground_truths

    dataset['column_names'] = list(dataset.keys())

    return DictWrapper(dataset)
