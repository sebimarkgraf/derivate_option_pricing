import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from option_pricing import BinomialModel, BlackScholesModel, Put, Underlying

plt.style.use("bmh")

underlying = Underlying(
    base_beginning=13144.28, volatility=0.2414, dividend=0.00, interest_rate=-0.00517
)
option = Put(underlying, strike_price=14000.0, american=False, T=0.5)

# Use the binomial model to calculate the option price
price, option_tree, stock_tree = BinomialModel(periods=100).calc_option_price(option)
print(f"b) Binomial Price bei n=100: {price / 100}")

# Aufgabe c)
prices = []
ns = []
# We walk through the logspace of periods to use less time.
# This gives equidistant points for the plotting later on
for n in tqdm(np.logspace(0, 3, num=100, dtype="int")):
    price, option_tree, stock_tree = BinomialModel(periods=n).calc_option_price(option)
    prices.append(price / 100)
    ns.append(n)

fix, ax = plt.subplots()
# Plot the binomial price curve against the periods
ax.plot(ns, prices, label="Binomial")

# Use the black scholes model to calculate the current option price
black_scholes = BlackScholesModel().calc_option_price(option=option)
print(f"Black Scholes Price: {black_scholes / 100}")

# Add the black scholes price as a red line into the plot
plt.axhline(black_scholes / 100, color="r", label="Black Scholes")

# Add the market price
plt.axhline(13.08, color="g", label="Markt")

# Plot styling
ax.set(
    xlabel="Binomial Perioden",
    ylabel="Option Preis (€)",
    title="Preis von DV7PCH",
    xscale="log",
)
ax.legend()
plt.savefig("price.png")


# f)
