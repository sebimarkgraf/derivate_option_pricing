from option_pricing import Underlying, BinomialModel, BlackScholesModel, Option

underlying = Underlying(base_beginning=13144.28, volatility=0.2414, dividend=0.00, interest_rate=-0.0517)
option = Option(underlying, strike_price=14000.0, is_call=False, american=False, T=0.5)


price, option_tree, stock_tree = BinomialModel(periods=100).calc_option_price(option)
print(f"Binomial Price: {price / 100}")
#print("Option")
#print(option_tree)
#print("Stock")
#print(stock_tree)
black_scholes = BlackScholesModel().calc_option_price(option=option)
print(f"Black Scholes Price: {black_scholes / 100}")
