import dataclasses
from abc import ABCMeta, abstractmethod

import numpy as np


@dataclasses.dataclass(frozen=True)
class Underlying:
    """
    Underlying allows to add different underlyings in an abstract way.
    This could be used for stocks or other values, as long
    as they have the required attributes of
    a beginning_price and volatility.
    """

    base_beginning: float
    volatility: float
    dividend: float
    interest_rate: float


@dataclasses.dataclass(frozen=True)
class Option(metaclass=ABCMeta):
    """
    An option created for an underlying.
    Mainly, implemented for calls (is_call=true"""

    underlying: Underlying
    T: float
    strike_price: float
    american: bool

    @abstractmethod
    def option_payoff(self, price):
        pass


class Call(Option):
    def option_payoff(self, price):
        return np.maximum(0, price - self.strike_price)


class Put(Option):
    def option_payoff(self, price):
        return np.maximum(0, self.strike_price - price)
