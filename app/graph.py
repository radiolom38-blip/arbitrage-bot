import networkx as nx
import logging
from config import Config

logger = logging.getLogger(__name__)

class GraphEngine:
    def __init__(self, config: Config):
        self.config = config
        self.graph = nx.DiGraph()

    def build_graph(self, pairs):
        self.graph.clear()
        for symbol, info in pairs.items():
            base, quote = info['baseCoin'], info['quoteCoin']  # Adjust based on Bybit symbol format
            # Add edge from quote to base (buy base with quote)
            price = self.get_price(symbol)  # From market.orderbooks
            if price:
                adj_price = price * (1 - self.config.FEE_RATE)
                self.graph.add_edge(quote, base, weight=-math.log(adj_price), symbol=symbol, direction='buy')
                # Add reverse edge (sell base for quote)
                rev_price = 1 / price * (1 - self.config.FEE_RATE)
                self.graph.add_edge(base, quote, weight=-math.log(rev_price), symbol=symbol, direction='sell')

    def get_price(self, symbol, direction='buy'):
        ob = market.orderbooks.get(symbol)  # Need market ref
        if direction == 'buy':
            return ob['ask']  # Buy at ask
        return ob['bid']  # Sell at bid

    # Note: To find cycles, use bellman-ford or similar for negative cycles, but since arbitrage, use log prices for multiplication