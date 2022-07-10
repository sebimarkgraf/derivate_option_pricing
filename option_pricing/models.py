import math

import numpy as np
import scipy.stats

from option_pricing.options import Call, Option, Put


class BinomialModel:
    """
    Implementation of Binomial model after Cox-Ross-Rubinstein.

    Simulates the price movement using the specified number of periods.
    Please note that an increasing number of periods leads to increased
    calculation time.
    """

    def __init__(self, periods: int):
        """
        Create the Binomial model with the specified number of periods.

        :param periods: number of periods to use in calculation
        """
        self.periods = periods

    def price_tree(self, price_beginning, up, down):
        """
        Create the tree of price at every movement

        :param price_beginning: beginning price
        :param up: fraction up movement
        :param down: fraction of down movement
        :return: resulting prices at all combinations
        """
        n = self.periods
        tree = np.zeros((n + 1, n + 1))
        for j in range(n + 1):
            for u in range(j + 1):
                tree[u][j] = price_beginning * (up**u) * (down ** (j - u))

        return tree

    def option_payoff_tree(self, stock_tree, option: Option):
        """
        Create payoff of option for every node in tree

        :param stock_tree: price of stock as tree. Created by price_tree
        :param option: the option to calculate the payoff
        :return: tree of option payoffs
        """
        return option.option_payoff(stock_tree)

    def recurse_option_tree(self, option_tree, option, pu, pd, r, div, delta_t):
        """
        Walk through the payoff tree and calculate the pricing of the option

        :param option_tree: payoffs of option from option_payoff_tree
        :param option: the option to calculate
        :param pu:
        :param pd:
        :param r:
        :param div:
        :param delta_t:
        :return:
        """
        for j in range(self.periods - 1, -1, -1):
            for i in range(j + 1):
                option_price = (
                    pu * option_tree[i + 1][j + 1] + pd * option_tree[i][j + 1]
                ) * np.exp(-1 * (r - div) * delta_t)
                # Use Early exercise price
                if option.american:
                    option_price = np.maximum(option_price, option_tree[i][j])

                option_tree[i][j] = option_price

        return option_tree

    def calc_option_price(self, option: Option):
        """
        Calculates the option pricing including the tree of prices at every movement

        :param option: the option to calculate
        :return:
        """
        T = option.T
        underlying = option.underlying
        div = underlying.dividend
        r = underlying.interest_rate
        delta_t = T / self.periods

        u = math.exp(underlying.volatility * math.sqrt(delta_t))
        d = 1 / u
        pu = (np.exp((r - div) * delta_t) - d) / (u - d)
        pd = 1 - pu
        stock_tree = self.price_tree(
            price_beginning=underlying.base_beginning, up=u, down=d
        )
        payoff_tree = self.option_payoff_tree(stock_tree, option)
        payoff_tree = self.recurse_option_tree(
            payoff_tree, option, pu, pd, r, div, delta_t
        )

        return payoff_tree[0][0], payoff_tree, stock_tree


class BlackScholesModel:
    """
    The Black-Scholes model, also known as the Black-Scholes-Merton (BSM) model,
    is one of the most important concepts in modern financial theory.
    This mathematical equation estimates the theoretical value of derivatives
    based on other investment instruments,
    taking into account the impact of time and other risk factors.
    Developed in 1973, it is still regarded as one of the best ways for pricing an options contract.
    """

    def calc_option_price(self, option: Option):
        T = option.T
        underlying = option.underlying
        d1 = (
            np.log(underlying.base_beginning / option.strike_price)
            + (
                (
                    (underlying.interest_rate - underlying.dividend)
                    + 0.5 * underlying.volatility**2
                )
                * T
            )
        ) / (
            underlying.volatility * np.sqrt(T)
        )  # d1
        d2 = (
            np.log(underlying.base_beginning / option.strike_price)
            + (
                (
                    (underlying.interest_rate - underlying.dividend)
                    - 0.5 * underlying.volatility**2
                )
                * T
            )
        ) / (
            underlying.volatility * np.sqrt(T)
        )  # d2
        Nd1 = scipy.stats.norm.cdf(d1, 0.0, 1.0)  # N(d1)
        Nd2 = scipy.stats.norm.cdf(d2, 0.0, 1.0)  # N(d1)
        Nminusd1 = scipy.stats.norm.cdf(-d1, 0.0, 1.0)  # N(-d1)
        Nminusd2 = scipy.stats.norm.cdf(-d2, 0.0, 1.0)  # N(-d2)

        if isinstance(option, Call):
            option_price = (
                underlying.base_beginning * np.exp(-underlying.dividend * T) * Nd1
                - option.strike_price * np.exp(-underlying.interest_rate * T) * Nd2
            )
        elif isinstance(option, Put):
            option_price = (
                option.strike_price * np.exp(-underlying.interest_rate * T) * Nminusd2
                - underlying.base_beginning
                * np.exp(-underlying.dividend * T)
                * Nminusd1
            )
        else:
            raise RuntimeError(f"Black Scholes not implemented for {type(option)}.")

        return option_price
