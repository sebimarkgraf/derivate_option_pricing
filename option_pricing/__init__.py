import dataclasses
import math
import numpy as np
import scipy.stats

@dataclasses.dataclass
class Underlying:
    base_beginning: float
    volatility: float
    dividend: float
    interest_rate: float


@dataclasses.dataclass
class Option:
    underlying: Underlying
    T: float
    strike_price: float
    is_call: bool
    american: bool


class BinomialModel:
    def __init__(self, periods: int):
        self.periods = periods

    def price_tree(self, price_beginning, up, down):
        n = self.periods
        tree = np.zeros((n + 1, n + 1))
        for j in range(n + 1):
            for u in range(j + 1):
                tree[u][j] = price_beginning * (up ** u) * (down ** (j - u))

        return tree

    def option_payoff_tree(self, stock_tree, option: Option):
        iopt = 1 if option.is_call else -1
        return np.maximum(0, iopt * (stock_tree - option.strike_price))

    def recurse_option_tree(self, option_tree, option, pu, pd, r, div, delta_t):
        for j in range(self.periods - 1, -1, -1):
            for i in range(j + 1):
                option_price = (pu * option_tree[i + 1][j + 1] + pd * option_tree[i][j + 1]) * np.exp(
                    -1 * (r - div) * delta_t)
                # Use Early exercise price
                if option.american:
                    option_price = np.maximum(option_price, option_tree[i][j])

                option_tree[i][j] = option_price

        return option_tree



    def calc_option_price(self, option: Option):
        T = option.T
        underlying = option.underlying
        div = underlying.dividend
        r = underlying.interest_rate
        delta_t = T / self.periods

        u = math.exp(underlying.volatility * math.sqrt(delta_t))
        d = 1 / u
        pu = (np.exp((r - div) * delta_t) - d) / (u - d)
        pd = 1 - pu
        stock_tree = self.price_tree(price_beginning=underlying.base_beginning, up=u, down=d)
        payoff_tree = self.option_payoff_tree(stock_tree, option)
        payoff_tree = self.recurse_option_tree(payoff_tree, option, pu, pd, r, div, delta_t)

        return payoff_tree[0][0], payoff_tree, stock_tree



class BlackScholesModel:
    def calc_option_price(self, option: Option):
        T = option.T
        underlying = option.underlying
        d1 = (np.log(underlying.base_beginning / option.strike_price) + (((underlying.interest_rate - underlying.dividend) + 0.5 * underlying.volatility ** 2) * T)) \
             / (underlying.volatility * np.sqrt(T))  # d1
        d2 = (np.log(underlying.base_beginning / option.strike_price) + (((underlying.interest_rate - underlying.dividend)
             - 0.5 * underlying.volatility ** 2) * T))\
             / (underlying.volatility * np.sqrt(T))  # d2
        Nd1 = scipy.stats.norm.cdf(d1, 0.0, 1.0)  # N(d1)
        Nd2 = scipy.stats.norm.cdf(d2, 0.0, 1.0)  # N(d1)
        Nminusd1 = scipy.stats.norm.cdf(-d1, 0.0, 1.0)  # N(-d1)
        Nminusd2 = scipy.stats.norm.cdf(-d2, 0.0, 1.0)  # N(-d2)

        if option.is_call:
            option_price = underlying.base_beginning * np.exp(-underlying.dividend * T) * Nd1 - \
                                 option.strike_price * np.exp(-underlying.interest_rate * T) * Nd2
        else:
            option_price = option.strike_price * np.exp(-underlying.interest_rate * T) * Nminusd2 - \
                                 underlying.base_beginning * np.exp(-underlying.dividend * T) * Nminusd1

        return option_price
