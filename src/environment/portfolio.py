""" portfolio.py

Represent the trader's portfolio, providing functions to
buy, sell and update the portfolio with time.
"""

class Portfolio:
    def __init__(self, startCash, tradeSize, batchIterator):
        """Initialize the portfolio

        Arguments:
            startCash {float} -- Starting cash for the trader
            tradeSize {float} -- Size of a single trade step
            batchIterator {BatchIterator} -- Price data iterator
        """

        self.cash = startCash
        self.trade_size = tradeSize
        self.batch_iterator = batchIterator
        self.secondary = 0.0
        self.price_data = next(self.batch_iterator)

    def price(self, time):
        """Get the secondary price at a given time

        Arguments:
            time {datetime} -- timestamp of the tick

        Returns:
            float -- price
        """

        price = 0.0
        try:
            price = self.price_data.loc[time]['close']
        except KeyError:
            self.price_data = next(self.batch_iterator)
            price = self.price_data.loc[time]['close']

        return price

    def value(self, time):
        """Portfolio value

        Arguments:
            time {datetime} -- timestamp of the tick

        Returns:
            float -- portfolio value
        """

        return self.cash + self.secondary * self.price(time)

    def buy(self, time):
        """Try to buy at given time

        Arguments:
            time {datetime} -- timestamp of the tick

        Returns:
            bool -- Success or failure
        """

        cost = self.price(time) * self.trade_size

        if cost > self.cash:
            return False

        self.cash -= cost
        self.secondary += self.trade_size

        return True

    def sell(self, time):
        """Try to sell at given time

        Arguments:
            time {datetime} -- timestamp of the tick

        Returns:
            bool -- Success or failure
        """

        if self.secondary < self.trade_size:
            return False

        cost = self.price(time) * self.trade_size

        self.secondary -= self.trade_size
        self.cash += cost

        return True

    def __str__(self):
        string_rep = "Portfolio\n"
        string_rep += "Cash: " + str(self.cash) + "\n"
        string_rep += "Secondary: " + str(self.secondary) + "\n"

        return string_rep


