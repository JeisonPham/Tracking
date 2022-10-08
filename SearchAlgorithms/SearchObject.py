from abc import ABC, abstractmethod


class SearchObject(ABC):

    @abstractmethod
    def get_next_searches(self):
        pass

    @abstractmethod
    def calc_heuristic(self, *args, **kwargs):
        pass

    @abstractmethod
    def calc_cost(self, *args, **kwargs):
        pass

    @abstractmethod
    def __eq__(self, other):
        pass

    @abstractmethod
    def __hash__(self):
        pass