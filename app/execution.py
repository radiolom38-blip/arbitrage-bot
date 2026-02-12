import asyncio
import time
import logging
from pybit.unified_trading import HTTP
from config import Config
from market import MarketData

logger = logging.getLogger(__name__)

class ExecutionEngine:
    def __init__(self, config: Config, market: MarketData):
        self.config = config
        self.market = market
        self.http = HTTP(
            testnet=config.DEMO_MODE,
            api_key=config.API_KEY,
            api_secret=config.API_SECRET
        )

    async def execute_cycle(self, slot, cycle):
        start_time = time.time()
        amount = self.config.CYCLE_SIZE
        current_asset = cycle[0]
        try:
            for i in range(len(cycle) - 1):
                from_asset, to_asset = current_asset, cycle[i+1]
                symbol = self.get_symbol(from_asset, to_asset)
                direction = self.get_direction(from_asset, to_asset)
                if direction == 'buy':
                    order = self.http.place_order(
                        category="spot",
                        symbol=symbol,
                        side="Buy",
                        orderType="Market",
                        qty=amount / price  # Need price
                    )
                else:
                    order = self.http.place_order(
                        category="spot",
                        symbol=symbol,
                        side="Sell",
                        orderType="Market",
                        qty=amount
                    )
                await self.wait_for_execution(order['orderId'])
                amount = self.get_filled_amount(order)  # Placeholder
                current_asset = to_asset
            
            duration = time.time() - start_time
            if duration > self.config.MAX_CYCLE_TIME:
                raise TimeoutError("Cycle timeout")
            logger.info(f"Cycle completed: {cycle} in {duration}s")
        except Exception as e:
            logger.error(f"Execution error: {e}")
            # Handle rollback
        finally:
            slot.release()

    async def wait_for_execution(self, order_id):
        for _ in range(10):
            status = self.http.get_order_history(category="spot", orderId=order_id)
            if status['status'] == 'Filled':
                return
            await asyncio.sleep(0.5)
        raise Exception("Order not filled")

    def get_symbol(self, from_asset, to_asset):
        # Logic to find symbol like BTCUSDT
        return f"{from_asset}{to_asset}".upper()

    def get_direction(self, from_asset, to_asset):
        # If selling from_asset for to_asset, etc.
        return 'buy' if to_asset == base else 'sell'