import dataclasses
from abc import ABCMeta, abstractmethod

import matplotlib.pyplot as plt
import numpy as np


@dataclasses.dataclass()
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


@dataclasses.dataclass()
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

    def plot_payoff(self, price_range=None):
        if price_range is None:
            min_price = self.underlying.base_beginning * (
                1 - self.underlying.volatility
            )
            max_price = self.underlying.base_beginning * (
                1 + self.underlying.volatility
            )
            price_range = [min_price, max_price]

        prices = np.linspace(*price_range, num=200)
        fig, ax = plt.subplots()
        ax.plot(prices, self.option_payoff(prices))
        ax.set(title="Payoff", xlabel="Underlying Price", ylabel="Payout")

        return fig


class Call(Option):
    def option_payoff(self, price):
        return np.maximum(0, price - self.strike_price)


class Put(Option):
    def option_payoff(self, price):
        return np.maximum(0, self.strike_price - price)


@dataclasses.dataclass
class SprintCertificate(Option):
    cap: float
    factor: float

    def option_payoff(self, price):
        cap_payout = (self.cap - self.strike_price) * self.factor
        return (
            np.minimum(
                cap_payout,
                np.maximum(
                    price - self.strike_price, (price - self.strike_price) * self.factor
                ),
            )
            + price
        )
