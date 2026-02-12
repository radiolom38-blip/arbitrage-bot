from pybit.unified_trading import HTTP, WebSocket
import asyncio
import logging
from config import Config

logger = logging.getLogger(__name__)

class MarketData:
    def __init__(self, config: Config):
        self.config = config
        self.http = HTTP(
            testnet=config.DEMO_MODE,
            api_key=config.API_KEY,
            api_secret=config.API_SECRET
        )
        self.ws = None
        self.pairs = {}
        self.orderbooks = {}
        self.lock = asyncio.Lock()

    async def connect_ws(self):
        self.ws = WebSocket(testnet=self.config.DEMO_MODE, channel_type="spot")
        # Subscribe to orderbooks for relevant pairs
        # This needs to be called after fetching pairs

    async def update_market_data(self):
        async with self.lock:
            try:
                symbols = self.http.get_symbols(category="spot")['result']['list']
                self.pairs = {s['symbol']: s for s in symbols if self.filter_pair(s)}
                await self.subscribe_to_orderbooks()
            except Exception as e:
                logger.error(f"Error updating market data: {e}")

    def filter_pair(self, symbol_info):
        # Implement filtering logic
        volume_24h = float(symbol_info.get('turnover24h', 0))
        if volume_24h < self.config.MIN_VOLUME_24H:
            return False
        # Fetch spread and depth - may need separate call
        # For simplicity, assume we fetch orderbook later
        return True

    async def subscribe_to_orderbooks(self):
        if not self.ws:
            await self.connect_ws()
        for symbol in self.pairs:
            self.ws.orderbook_stream(50, symbol, self.orderbook_callback)

    def orderbook_callback(self, data):
        symbol = data['s']
        self.orderbooks[symbol] = {
            'bid': float(data['b'][0][0]) if data['b'] else None,
            'ask': float(data['a'][0][0]) if data['a'] else None,
            # Add depth calculation
        }

    async def close(self):
        if self.ws:
            self.ws.exit()