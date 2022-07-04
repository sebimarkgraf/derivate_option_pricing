from option_pricing import Underlying, BinomialModel, BlackScholesModel, Option
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
plt.style.use("bmh")

underlying = Underlying(base_beginning=13144.28, volatility=0.2414, dividend=0.00, interest_rate=-0.0517)
option = Option(underlying, strike_price=14000.0, is_call=False, american=False, T=0.5)


price, option_tree, stock_tree = BinomialModel(periods=100).calc_option_price(option)
print(f"b) Binomial Price bei n=100: {price / 100}")

## Aufgabe c)
prices = []
ns = []
for n in tqdm(np.logspace(0, 3, num=100, dtype='int')):
    price, option_tree, stock_tree = BinomialModel(periods=n).calc_option_price(option)
    prices.append(price / 100)
    ns.append(n)

fix, ax = plt.subplots()
ax.plot(ns, prices, label="Binomial")


black_scholes = BlackScholesModel().calc_option_price(option=option)
print(f"Black Scholes Price: {black_scholes / 100}")

plt.axhline(black_scholes / 100, color='r', label="Black Scholes")
plt.axhline(13.08, color='g', label="Markt")

ax.set(xlabel="Binomial Perioden", ylabel="Option Preis (€)", title="Preis von DV7PCH", xscale="log")
ax.legend()
plt.savefig('price.png')


# f)

