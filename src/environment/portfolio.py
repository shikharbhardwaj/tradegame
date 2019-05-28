""" portfolio.py

Represent the trader's portfolio, providing functions to
buy, sell and update the portfolio with time.
"""

class Portfolio:
    def __init__(self, startCash, tradeSize, batchIterator, spread=0.0):
        """Initialize the portfolio

        Arguments:
            startCash {float} -- Starting cash for the trader
            tradeSize {float} -- Size of a single trade step
            batchIterator {BatchIterator} -- Price data iterator
            spread {float} -- Fixed spread for fees calculation
        """

        self.cash = startCash
        self.trade_size = tradeSize
        self.batch_iterator = batchIterator
        self.spread = spread
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

    def valid_actions(self, time):
        actions = [0]

        cost = self.price(time) * self.trade_size

        if cost <= self.cash:
            actions.append(1)

        if self.secondary >= self.trade_size:
            actions.append(2)

        return actions

    def valueAtPrice(self, price):
        """Portfolio value at given price.

        Arguments:
            price {float} -- Price of secondary currency.

        Returns:
            float -- portfolio value
        """

        return self.cash + self.secondary * price

    def valueAtTime(self, time):
        """Portfolio value at given time.

        Arguments:
            time {datetime} -- timestamp of the tick

        Returns:
            float -- portfolio value
        """

        return self.valueAtPrice(self.price(time))

    def buyAtTime(self, time):
        """Try to buy at given time.

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

    def buyAtPrice(self, price):
        """Try to buy at given price.

        Arguments:
            price {float} -- price

        Returns:
            float -- Success or failure
        """

        cost = price * self.trade_size

        if cost > self.cash:
            return False

        self.cash -= cost
        self.secondary += self.trade_size

        return True

    def sellAtTime(self, time):
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

    def sellAtPrice(self, price):
        """Try to sell at given price.

        Arguments:
            time {float} -- price

        Returns:
            bool -- Success or failure
        """

        if self.secondary < self.trade_size:
            return False

        cost = price * self.trade_size

        self.secondary -= self.trade_size
        self.cash += cost

        return True

    def __str__(self):
        string_rep  = "======================\n"
        string_rep += "       Portfolio\n"
        string_rep += "======================\n"
        string_rep += "Cash: " + str(self.cash) + "\n"
        string_rep += "Secondary: " + str(self.secondary)

        return string_rep


