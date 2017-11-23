import abc

class Model(object, metaclass=abc.ABCMeta):
    def __init__(self):
        pass

    @abc.abstractmethod
    def train(self, X_df, Y):
        raise NotImplementedError('users must define train() to use this base class')

    @abc.abstractmethod
    def predict(self, X_df):
        raise NotImplementedError('users must define predict() to use this base class')
