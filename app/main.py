import asyncio
import logging
import os
import signal
from config import Config
from market import MarketData
from graph import GraphEngine
from arbitrage import ArbitrageEngine
from capital import CapitalController
from risk import RiskManager
from execution import ExecutionEngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ArbitrageBot:
    def __init__(self):
        self.config = Config()
        self.market = MarketData(self.config)
        self.graph = GraphEngine(self.config)
        self.arbitrage = ArbitrageEngine(self.config, self.graph)
        self.capital = CapitalController(self.config)
        self.risk = RiskManager(self.config, self.capital)
        self.execution = ExecutionEngine(self.config, self.market)
        self.bot_running = True

    async def main_loop(self):
        while self.bot_running:
            try:
                await self.market.update_market_data()
                self.graph.build_graph(self.market.pairs)
                profitable_cycles = self.arbitrage.find_profitable_cycles()
                
                for cycle in sorted(profitable_cycles, key=lambda x: x['expected_profit'] * x['liquidity_score'], reverse=True):
                    slot = self.capital.get_free_slot()
                    if slot:
                        if await self.risk.check_before_execution(cycle):
                            asyncio.create_task(self.execution.execute_cycle(slot, cycle))
                
                if self.risk.check_daily_loss():
                    logger.warning("Daily loss limit reached. Stopping bot.")
                    self.bot_running = False
                
                await asyncio.sleep(0.1)  # Small delay to prevent CPU overload
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(1)

    async def shutdown(self):
        self.bot_running = False
        await self.market.close()
        logger.info("Bot shutdown complete.")

def handle_shutdown(loop, bot):
    async def shutdown_handler():
        await bot.shutdown()
        loop.stop()
    asyncio.create_task(shutdown_handler())

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    bot = ArbitrageBot()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: handle_shutdown(loop, bot))
    try:
        loop.run_until_complete(bot.main_loop())
    finally:
        loop.close()