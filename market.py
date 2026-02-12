from pybit.unified_trading import HTTP, WebSocket
import asyncio
import logging
import time
from config import Config

logger = logging.getLogger(__name__)

class MarketData:
    def __init__(self, config: Config):
        self.config = config
        self.http = HTTP(
            testnet=False,
            api_key=config.API_KEY,
            api_secret=config.API_SECRET,
            base_url="https://api-demo.bybit.com",
            recv_window=10000
        )
        self.ws = None
        self.orderbooks = {}           # symbol → {'bids': list, 'asks': list, 'ts': float}
        self.symbols = []              # отфильтрованные символы
        self._running = False
        self._reconnect_delay = 5

    async def start_websocket(self):
        self._running = True
        asyncio.create_task(self._ws_loop())

    async def _ws_loop(self):
        while self._running:
            try:
                self.ws = WebSocket(
                    testnet=False,
                    channel_type="spot",
                    api_key=self.config.API_KEY if self.config.API_KEY else None,
                    api_secret=self.config.API_SECRET if self.config.API_SECRET else None,
                )

                def handle_orderbook(message):
                    if 'topic' not in message or 'data' not in message:
                        return
                    symbol = message['topic'].split('.')[1]
                    data = message['data']
                    ts = time.time()
                    self.orderbooks[symbol] = {
                        'bids': data.get('b', []),
                        'asks': data.get('a', []),
                        'ts': ts
                    }

                # Подписка на orderbook 50 levels для всех отфильтрованных символов
                for sym in self.symbols[:50]:  # лимит подписок, не перегружать
                    self.ws.orderbook_stream(50, sym, callback=handle_orderbook)

                # Heartbeat / keep-alive
                while self._running and self.ws.ws.sock and self.ws.ws.sock.connected:
                    await asyncio.sleep(20)
                    # можно отправить ping если нужно

            except Exception as e:
                logger.error(f"WS error: {e}. Reconnecting in {self._reconnect_delay}s")
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(self._reconnect_delay * 2, 60)

    async def update_pairs_and_filter(self):
        try:
            resp = self.http.get_instruments_info(category="spot")
            if resp['retCode'] != 0:
                logger.error(f"get_instruments_info error: {resp}")
                return

            candidates = []
            for instr in resp['result']['list']:
                symbol = instr['symbol']
                volume24h = float(instr.get('volume24h', 0))
                if volume24h < self.config.MIN_VOLUME_24H:
                    continue
                # Другие фильтры: status == 'Trading', etc.
                if instr.get('status') == 'Trading':
                    candidates.append(symbol)

            self.symbols = candidates[:100]  # лимит для начала
            logger.info(f"Filtered {len(self.symbols)} symbols")
        except Exception as e:
            logger.error(f"Update pairs error: {e}")

    def get_best_bid_ask(self, symbol):
        ob = self.orderbooks.get(symbol)
        if not ob:
            return None, None
        bids = ob['bids']
        asks = ob['asks']
        best_bid = float(bids[0][0]) if bids else None
        best_ask = float(asks[0][0]) if asks else None
        return best_bid, best_ask

    async def stop(self):
        self._running = False
        if self.ws:
            self.ws.exit()