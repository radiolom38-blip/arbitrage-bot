"""
Cloud-Optimized Arbitrage Bot with Telegram Notifications
Enhanced version for cloud deployment
"""

import os
import signal
import psutil
from arbitrage_bot import *


# ============================================================================
# CLOUD CONFIGURATION
# ============================================================================

class CloudConfig(Config):
    """Extended config for cloud deployment"""
    
    # Cloud detection
    IS_CLOUD = bool(os.getenv('RAILWAY_ENVIRONMENT') or os.getenv('RENDER'))
    ENABLE_DASHBOARD = os.getenv('ENABLE_DASHBOARD', 'false').lower() == 'true'
    
    # Telegram notifications
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
    
    # Cloud optimizations
    PAIR_FILTER_INTERVAL = 600  # 10 minutes (save API calls)
    METRICS_REPORT_INTERVAL = 3600  # 1 hour
    SYSTEM_METRICS_INTERVAL = 3600  # 1 hour


# ============================================================================
# TELEGRAM NOTIFICATIONS
# ============================================================================

class TelegramNotifier:
    """Send notifications via Telegram"""
    
    def __init__(self):
        self.token = CloudConfig.TELEGRAM_BOT_TOKEN
        self.chat_id = CloudConfig.TELEGRAM_CHAT_ID
        self.enabled = bool(self.token and self.chat_id)
        
        if self.enabled:
            logging.info("Telegram notifications enabled")
        else:
            logging.info("Telegram notifications disabled (no credentials)")
    
    def send(self, message: str, silent: bool = False):
        """Send message to Telegram"""
        if not self.enabled:
            return
        
        try:
            import requests
            
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            data = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "HTML",
                "disable_notification": silent
            }
            
            response = requests.post(url, data=data, timeout=5)
            if response.status_code != 200:
                logging.error(f"Telegram API error: {response.text}")
        
        except Exception as e:
            logging.error(f"Failed to send Telegram message: {e}")
    
    def send_startup(self):
        """Send startup notification"""
        self.send(
            "🚀 <b>Arbitrage Bot Started</b>\n"
            f"Environment: {'Cloud' if CloudConfig.IS_CLOUD else 'Local'}\n"
            f"Capital: {Config.MAX_CAPITAL} USDT\n"
            f"Max Cycles: {Config.MAX_ACTIVE_CYCLES}"
        )
    
    def send_cycle_complete(self, slot_id: int, profit: float, duration: float, cycle_path: str):
        """Send cycle completion notification"""
        emoji = "✅" if profit > 0 else "❌"
        self.send(
            f"{emoji} <b>Cycle #{slot_id}</b>\n"
            f"Path: {cycle_path}\n"
            f"Profit: {profit:.4f} USDT ({profit/Config.CYCLE_SIZE*100:.2f}%)\n"
            f"Time: {duration:.2f}s",
            silent=(profit < 0)  # Silent for losses
        )
    
    def send_status_report(self, metrics: BotMetrics):
        """Send periodic status report"""
        self.send(
            "📊 <b>Status Report</b>\n"
            f"Total Cycles: {metrics.total_cycles}\n"
            f"Win Rate: {metrics.win_rate*100:.1f}%\n"
            f"Total Profit: {metrics.total_profit:.2f} USDT\n"
            f"Avg Profit/Cycle: {metrics.avg_profit_per_cycle:.4f} USDT\n"
            f"Active Slots: {metrics.active_slots}/{Config.MAX_ACTIVE_CYCLES}",
            silent=True
        )
    
    def send_system_metrics(self):
        """Send system resource metrics"""
        try:
            metrics = {
                'cpu': psutil.cpu_percent(interval=1),
                'ram': psutil.virtual_memory().percent,
                'ram_mb': psutil.virtual_memory().used / 1024 / 1024,
                'disk': psutil.disk_usage('/').percent
            }
            
            self.send(
                "💻 <b>System Metrics</b>\n"
                f"CPU: {metrics['cpu']:.1f}%\n"
                f"RAM: {metrics['ram']:.1f}% ({metrics['ram_mb']:.0f} MB)\n"
                f"Disk: {metrics['disk']:.1f}%",
                silent=True
            )
        except Exception as e:
            logging.error(f"Failed to get system metrics: {e}")
    
    def send_error(self, error_msg: str):
        """Send error notification"""
        self.send(
            f"🚨 <b>Error</b>\n"
            f"{error_msg}"
        )
    
    def send_shutdown(self, reason: str = "Normal"):
        """Send shutdown notification"""
        self.send(
            f"🛑 <b>Bot Shutdown</b>\n"
            f"Reason: {reason}"
        )


# ============================================================================
# ENHANCED ARBITRAGE BOT FOR CLOUD
# ============================================================================

class CloudArbitrageBot(ArbitrageBot):
    """Enhanced bot with cloud features"""
    
    def __init__(self, api_key: str, api_secret: str):
        super().__init__(api_key, api_secret)
        
        # Telegram notifier
        self.notifier = TelegramNotifier()
        
        # Shutdown handling
        self.shutdown_requested = False
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)
        
        # Metrics tracking
        self.last_metrics_report = 0
        self.last_system_metrics = 0
        
        logging.info("CloudArbitrageBot initialized")
    
    def _handle_shutdown(self, signum, frame):
        """Handle graceful shutdown"""
        logging.info(f"Shutdown signal {signum} received")
        self.shutdown_requested = True
        self.risk_manager.bot_active = False
        self.notifier.send_shutdown("Signal received")
    
    async def start(self):
        """Start bot with cloud enhancements"""
        try:
            logging.info("Starting Cloud Arbitrage Bot...")
            self.notifier.send_startup()
            
            # Connect WebSocket
            await self.market_data.connect()
            
            # Initial pair filtering
            self.active_pairs = await self.pair_filter.update_pairs()
            
            # Subscribe to orderbooks
            symbols = [p.symbol for p in self.active_pairs]
            await self.market_data.subscribe_orderbook(symbols)
            
            # Wait for data
            await asyncio.sleep(3)
            
            # Start main loop
            await self.main_loop()
            
        except Exception as e:
            error_msg = f"Fatal error: {e}"
            logging.error(error_msg)
            self.notifier.send_error(error_msg)
            raise
    
    async def main_loop(self):
        """Enhanced main loop with notifications"""
        logging.info("Entering main loop...")
        
        while self.risk_manager.is_bot_active() and not self.shutdown_requested:
            try:
                # Update pairs periodically
                if time.time() - self.last_pair_update > CloudConfig.PAIR_FILTER_INTERVAL:
                    self.active_pairs = await self.pair_filter.update_pairs()
                    symbols = [p.symbol for p in self.active_pairs]
                    await self.market_data.subscribe_orderbook(symbols)
                    self.last_pair_update = time.time()
                
                # Build graph
                orderbooks = self.market_data.get_all_orderbooks()
                self.graph_builder.build_graph(self.active_pairs, orderbooks)
                
                # Find cycles
                cycles = self.arbitrage_engine.find_cycles()
                
                # Execute cycles
                for cycle in cycles[:10]:
                    slot = await self.capital_controller.get_free_slot()
                    if not slot:
                        break
                    
                    is_valid, liquidity_score = await self.liquidity_validator.validate_cycle(cycle)
                    if not is_valid:
                        continue
                    
                    cycle.liquidity_score = liquidity_score
                    
                    allocated = await self.capital_controller.allocate_slot(slot, cycle)
                    if not allocated:
                        break
                    
                    asyncio.create_task(self._execute_slot(slot))
                    
                    logging.info(
                        f"Started cycle in slot {slot.slot_id}: "
                        f"{' -> '.join(cycle.path)} "
                        f"(expected: {cycle.expected_profit*100:.3f}%)"
                    )
                
                # Update metrics
                self.metrics.active_slots = self.capital_controller.get_active_count()
                
                # Periodic reports
                if time.time() - self.last_metrics_report > CloudConfig.METRICS_REPORT_INTERVAL:
                    self.notifier.send_status_report(self.metrics)
                    self.last_metrics_report = time.time()
                
                # System metrics
                if time.time() - self.last_system_metrics > CloudConfig.SYSTEM_METRICS_INTERVAL:
                    self.notifier.send_system_metrics()
                    self.last_system_metrics = time.time()
                
                # Log status
                if self.metrics.total_cycles % 10 == 0 and self.metrics.total_cycles > 0:
                    self._log_status()
                
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logging.error(f"Error in main loop: {e}")
                await asyncio.sleep(1)
        
        # Wait for active cycles to complete
        logging.info("Waiting for active cycles to complete...")
        while self.capital_controller.get_active_count() > 0:
            await asyncio.sleep(1)
        
        logging.info("Bot stopped gracefully")
    
    async def _execute_slot(self, slot: ExecutionSlot):
        """Execute with notifications"""
        start_time = time.time()
        
        try:
            success = await self.execution_engine.execute_cycle(slot)
            duration = time.time() - start_time
            
            profit = slot.actual_profit if success else -Config.CYCLE_SIZE * Config.EMERGENCY_STOP_LOSS
            self.metrics.record_cycle(success, profit, duration)
            
            # Send notification
            cycle_path = ' -> '.join(slot.cycle.path) if slot.cycle else 'Unknown'
            self.notifier.send_cycle_complete(slot.slot_id, profit, duration, cycle_path)
            
            status = SlotStatus.COMPLETED if success else SlotStatus.FAILED
            await self.capital_controller.release_slot(slot, status)
            
            if success:
                logging.info(
                    f"Slot {slot.slot_id} COMPLETED: "
                    f"profit={profit:.4f} USDT ({profit/Config.CYCLE_SIZE*100:.3f}%) "
                    f"time={duration:.2f}s"
                )
            else:
                logging.warning(f"Slot {slot.slot_id} FAILED: {slot.error_message}")
                
        except Exception as e:
            logging.error(f"Error executing slot {slot.slot_id}: {e}")
            self.notifier.send_error(f"Slot {slot.slot_id} error: {e}")
            await self.capital_controller.release_slot(slot, SlotStatus.FAILED)


# ============================================================================
# MAIN WITH AUTO-RESTART
# ============================================================================

async def main():
    """Main with auto-restart capability"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler('arbitrage_bot.log'),
            logging.StreamHandler()
        ]
    )
    
    # Load credentials
    api_key = os.getenv('BYBIT_API_KEY')
    api_secret = os.getenv('BYBIT_API_SECRET')
    
    if not api_key or not api_secret:
        logging.error("Missing BYBIT_API_KEY or BYBIT_API_SECRET")
        return
    
    # Auto-restart logic
    MAX_RESTARTS = 5
    restart_count = 0
    
    while restart_count < MAX_RESTARTS:
        try:
            # Create bot
            bot = CloudArbitrageBot(api_key, api_secret)
            
            # Start bot (with or without dashboard)
            if CloudConfig.ENABLE_DASHBOARD and not CloudConfig.IS_CLOUD:
                dashboard = WebDashboard(bot, port=8080)
                await asyncio.gather(bot.start(), dashboard.start())
            else:
                await bot.start()
            
            break  # Normal exit
            
        except KeyboardInterrupt:
            logging.info("Keyboard interrupt, shutting down...")
            break
            
        except Exception as e:
            restart_count += 1
            logging.error(
                f"Critical error (restart {restart_count}/{MAX_RESTARTS}): {e}",
                exc_info=True
            )
            
            if restart_count < MAX_RESTARTS:
                wait_time = min(30 * restart_count, 300)  # Max 5 min
                logging.info(f"Restarting in {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                logging.critical("Max restarts reached, exiting")
                
                # Send critical alert
                notifier = TelegramNotifier()
                notifier.send_error("🚨 Bot crashed after max restarts!")


if __name__ == "__main__":
    asyncio.run(main())
