import asyncio
import logging
import signal
from config import Config
from market import MarketData
from graph import GraphEngine
from arbitrage import ArbitrageEngine
from capital import CapitalController
from risk import RiskManager
from execution import ExecutionEngine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class ArbitrageBot:
    def __init__(self):
        self.config = Config()
        self.market = MarketData(self.config)
        self.graph_engine = GraphEngine(self.config, self.market)
        self.arbitrage = ArbitrageEngine(self.config, self.graph_engine)
        self.capital = CapitalController(self.config)
        self.risk = RiskManager(self.config, self.capital)
        self.execution = ExecutionEngine(self.config, self.market)
        self.running = True

    async def run(self):
        await self.market.start_websocket()
        while self.running:
            try:
                await self.market.update_pairs_and_filter()
                self.graph_engine.build_graph()
                cycles = self.arbitrage.find_profitable_cycles()
                for cycle_info in sorted(cycles, key=lambda x: x['score'], reverse=True):
                    slot = self.capital.get_free_slot()
                    if slot and await self.risk.can_execute(cycle_info):
                        asyncio.create_task(
                            self.execution.execute_cycle(slot, cycle_info['cycle'])
                        )
                await self.risk.check_limits()
                await asyncio.sleep(0.2)
            except Exception as e:
                logger.exception(f"Main loop error: {e}")
                await asyncio.sleep(2)

    async def shutdown(self):
        self.running = False
        await self.market.stop()
        logger.info("Bot stopped gracefully.")

async def main():
    bot = ArbitrageBot()
    loop = asyncio.get_running_loop()

    def handle_shutdown():
        asyncio.create_task(bot.shutdown())

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, handle_shutdown)

    try:
        await bot.run()
    except KeyboardInterrupt:
        pass
    finally:
        await bot.shutdown()

if __name__ == "__main__":
    asyncio.run(main())