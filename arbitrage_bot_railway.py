"""
Arbitrage Bot for Bybit DEMO Account - Railway Cloud Deployment
Standalone version with Telegram notifications
"""

import asyncio
import logging
import math
import time
import os
import signal
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple
import json

# Optional imports
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

from pybit.unified_trading import WebSocket, HTTP
import networkx as nx


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Bot configuration for Bybit DEMO account"""
    
    # ===== BYBIT DEMO SETTINGS =====
    # CRITICAL: Use DEMO endpoints, not testnet!
    DEMO_MODE = True
    REST_ENDPOINT = "https://api-demo.bybit.com"
    WS_ENDPOINT = "wss://stream-demo.bybit.com/v5/public/spot"
    
    # Capital Management
    MAX_CAPITAL = 333.0
    WORKING_CAPITAL = 300.0
    RESERVE = 33.0
    CYCLE_SIZE = 50.0
    MAX_ACTIVE_CYCLES = 6
    
    # Profit & Risk
    MIN_EXPECTED_PROFIT = 0.0015  # 0.15%
    DAILY_LOSS_LIMIT = 0.03  # 3%
    MAX_CYCLE_TIME = 5.0  # seconds
    EMERGENCY_STOP_LOSS = 0.003  # 0.3%
    
    # Pair Filtering
    MIN_24H_VOLUME = 10000  # USDT
    MAX_SPREAD = 0.002  # 0.2%
    MIN_DEPTH_USDT = 100
    
    # Trading
    BYBIT_FEE = 0.001  # 0.1%
    SLIPPAGE_FACTOR = 0.0005  # 0.05%
    
    # Graph Search
    START_CURRENCIES = ['USDT', 'USDC', 'BTC', 'ETH']
    MAX_CYCLE_LENGTH = 4
    
    # Update Intervals
    PAIR_FILTER_INTERVAL = 600  # 10 minutes (cloud optimization)
    
    # Cloud Settings
    IS_CLOUD = bool(os.getenv('RAILWAY_ENVIRONMENT') or os.getenv('RENDER'))
    ENABLE_DASHBOARD = os.getenv('ENABLE_DASHBOARD', 'false').lower() == 'true'
    
    # Telegram
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
    METRICS_REPORT_INTERVAL = 3600  # 1 hour
    SYSTEM_METRICS_INTERVAL = 3600  # 1 hour


# ============================================================================
# DATA CLASSES
# ============================================================================

class SlotStatus(Enum):
    IDLE = "IDLE"
    EXECUTING = "EXECUTING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class OrderBook:
    symbol: str
    bids: List[Tuple[float, float]] = field(default_factory=list)
    asks: List[Tuple[float, float]] = field(default_factory=list)
    timestamp: float = 0.0
    
    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0][0] if self.bids else None
    
    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0][0] if self.asks else None
    
    @property
    def spread(self) -> float:
        if self.best_bid and self.best_ask:
            return (self.best_ask - self.best_bid) / self.best_bid
        return float('inf')


@dataclass
class TradingPair:
    symbol: str
    base_currency: str
    quote_currency: str
    volume_24h: float = 0.0
    status: str = "Trading"


@dataclass
class ArbitrageCycle:
    path: List[str]
    pairs: List[str]
    sides: List[OrderSide]
    expected_profit: float
    expected_return: float
    liquidity_score: float = 1.0
    start_amount: float = Config.CYCLE_SIZE


@dataclass
class ExecutionSlot:
    slot_id: int
    status: SlotStatus = SlotStatus.IDLE
    capital: float = Config.CYCLE_SIZE
    cycle: Optional[ArbitrageCycle] = None
    start_time: float = 0.0
    executed_orders: List[Dict] = field(default_factory=list)
    current_step: int = 0
    actual_profit: float = 0.0
    error_message: str = ""


@dataclass
class BotMetrics:
    total_cycles: int = 0
    successful_cycles: int = 0
    failed_cycles: int = 0
    total_profit: float = 0.0
    daily_profit: float = 0.0
    daily_loss: float = 0.0
    win_rate: float = 0.0
    avg_profit_per_cycle: float = 0.0
    avg_cycle_time: float = 0.0
    total_cycle_time: float = 0.0
    active_slots: int = 0
    recent_cycles: deque = field(default_factory=lambda: deque(maxlen=100))
    start_time: float = field(default_factory=time.time)
    
    def update_daily_pnl(self, profit: float):
        if profit > 0:
            self.daily_profit += profit
        else:
            self.daily_loss += abs(profit)
    
    def record_cycle(self, success: bool, profit: float, duration: float):
        self.total_cycles += 1
        if success:
            self.successful_cycles += 1
        else:
            self.failed_cycles += 1
        
        self.total_profit += profit
        self.update_daily_pnl(profit)
        self.total_cycle_time += duration
        
        self.win_rate = self.successful_cycles / self.total_cycles if self.total_cycles > 0 else 0
        self.avg_profit_per_cycle = self.total_profit / self.total_cycles if self.total_cycles > 0 else 0
        self.avg_cycle_time = self.total_cycle_time / self.total_cycles if self.total_cycles > 0 else 0
        
        self.recent_cycles.append({
            'timestamp': time.time(),
            'success': success,
            'profit': profit,
            'duration': duration
        })


# ============================================================================
# TELEGRAM NOTIFICATIONS
# ============================================================================

class TelegramNotifier:
    """Send notifications via Telegram"""
    
    def __init__(self):
        self.token = Config.TELEGRAM_BOT_TOKEN
        self.chat_id = Config.TELEGRAM_CHAT_ID
        self.enabled = bool(self.token and self.chat_id and REQUESTS_AVAILABLE)
        
        if self.enabled:
            logging.info("✅ Telegram notifications ENABLED")
        else:
            reason = "no credentials" if not (self.token and self.chat_id) else "requests not installed"
            logging.info(f"❌ Telegram notifications DISABLED ({reason})")
    
    def send(self, message: str, silent: bool = False):
        """Send message to Telegram"""
        if not self.enabled:
            return
        
        try:
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
            logging.error(f"Failed to send Telegram: {e}")
    
    def send_startup(self):
        """Send startup notification"""
        self.send(
            "🚀 <b>Arbitrage Bot Started</b>\n"
            f"<b>Platform:</b> {'Railway Cloud ☁️' if Config.IS_CLOUD else 'Local'}\n"
            f"<b>Bybit Mode:</b> DEMO Account 🧪\n"
            f"<b>Capital:</b> {Config.MAX_CAPITAL} USDT\n"
            f"<b>Max Cycles:</b> {Config.MAX_ACTIVE_CYCLES}\n"
            f"<b>Min Profit:</b> {Config.MIN_EXPECTED_PROFIT*100:.2f}%"
        )
    
    def send_cycle_complete(self, slot_id: int, profit: float, duration: float, cycle_path: str):
        """Send cycle completion"""
        emoji = "✅" if profit > 0 else "❌"
        profit_pct = profit / Config.CYCLE_SIZE * 100
        self.send(
            f"{emoji} <b>Cycle #{slot_id}</b>\n"
            f"<b>Path:</b> {cycle_path}\n"
            f"<b>Profit:</b> {profit:.4f} USDT ({profit_pct:+.2f}%)\n"
            f"<b>Time:</b> {duration:.2f}s",
            silent=(profit < 0)
        )
    
    def send_status_report(self, metrics: BotMetrics):
        """Send periodic status"""
        self.send(
            "📊 <b>Status Report</b>\n"
            f"<b>Total Cycles:</b> {metrics.total_cycles}\n"
            f"<b>Win Rate:</b> {metrics.win_rate*100:.1f}%\n"
            f"<b>Total Profit:</b> {metrics.total_profit:.2f} USDT\n"
            f"<b>Avg/Cycle:</b> {metrics.avg_profit_per_cycle:.4f} USDT\n"
            f"<b>Active:</b> {metrics.active_slots}/{Config.MAX_ACTIVE_CYCLES}",
            silent=True
        )
    
    def send_system_metrics(self):
        """Send system metrics"""
        if not PSUTIL_AVAILABLE:
            return
        
        try:
            cpu = psutil.cpu_percent(interval=1)
            ram = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            self.send(
                "💻 <b>System Metrics</b>\n"
                f"<b>CPU:</b> {cpu:.1f}%\n"
                f"<b>RAM:</b> {ram.percent:.1f}% ({ram.used/1024/1024:.0f} MB)\n"
                f"<b>Disk:</b> {disk.percent:.1f}%",
                silent=True
            )
        except Exception as e:
            logging.error(f"System metrics error: {e}")
    
    def send_error(self, error_msg: str):
        """Send error alert"""
        self.send(f"🚨 <b>Error</b>\n{error_msg}")
    
    def send_shutdown(self, reason: str = "Normal"):
        """Send shutdown notification"""
        self.send(f"🛑 <b>Bot Shutdown</b>\n<b>Reason:</b> {reason}")


# ============================================================================
# MARKET DATA MODULE
# ============================================================================

class MarketDataModule:
    def __init__(self):
        self.orderbooks: Dict[str, OrderBook] = {}
        self.ws = None
        self.subscribed_symbols: Set[str] = set()
        self.lock = asyncio.Lock()
        
    async def connect(self):
        """Connect to Bybit DEMO WebSocket"""
        try:
            # Use DEMO WebSocket endpoint
            self.ws = WebSocket(
                testnet=False,  # Not testnet!
                demo=True,      # DEMO mode
                channel_type="spot"
            )
            
            logging.info(f"✅ WebSocket connected to DEMO: {Config.WS_ENDPOINT}")
        except Exception as e:
            logging.error(f"WebSocket connection error: {e}")
            raise
    
    async def subscribe_orderbook(self, symbols: List[str]):
        """Subscribe to orderbooks"""
        for symbol in symbols:
            if symbol not in self.subscribed_symbols:
                try:
                    self.ws.orderbook_stream(
                        depth=50,
                        symbol=symbol,
                        callback=self._handle_orderbook_update
                    )
                    self.subscribed_symbols.add(symbol)
                    
                    async with self.lock:
                        if symbol not in self.orderbooks:
                            self.orderbooks[symbol] = OrderBook(symbol=symbol)
                    
                    logging.info(f"📊 Subscribed: {symbol}")
                except Exception as e:
                    logging.error(f"Failed to subscribe {symbol}: {e}")
    
    def _handle_orderbook_update(self, message):
        """Handle orderbook updates"""
        try:
            if message.get('topic') and 'orderbook' in message['topic']:
                data = message.get('data', {})
                symbol = message['topic'].split('.')[-1]
                
                bids = [(float(p), float(q)) for p, q in data.get('b', [])][:10]
                asks = [(float(p), float(q)) for p, q in data.get('a', [])][:10]
                
                if symbol in self.orderbooks:
                    orderbook = self.orderbooks[symbol]
                    if bids:
                        orderbook.bids = sorted(bids, key=lambda x: x[0], reverse=True)
                    if asks:
                        orderbook.asks = sorted(asks, key=lambda x: x[0])
                    orderbook.timestamp = time.time()
        except Exception as e:
            logging.error(f"Orderbook update error: {e}")
    
    async def get_orderbook(self, symbol: str) -> Optional[OrderBook]:
        async with self.lock:
            return self.orderbooks.get(symbol)
    
    def get_all_orderbooks(self) -> Dict[str, OrderBook]:
        return self.orderbooks.copy()


# ============================================================================
# PAIR FILTER MODULE
# ============================================================================

class PairFilterModule:
    def __init__(self, http_client: HTTP):
        self.http = http_client
        self.active_pairs: List[TradingPair] = []
        self.last_update = 0.0
        
    async def update_pairs(self) -> List[TradingPair]:
        """Filter trading pairs"""
        try:
            response = self.http.get_instruments_info(category="spot")
            
            if response['retCode'] != 0:
                logging.error(f"Failed to get instruments: {response['retMsg']}")
                return self.active_pairs
            
            filtered_pairs = []
            instruments = response['result']['list']
            
            for inst in instruments:
                symbol = inst['symbol']
                
                ticker = self.http.get_tickers(category="spot", symbol=symbol)
                
                if ticker['retCode'] != 0:
                    continue
                
                ticker_data = ticker['result']['list'][0] if ticker['result']['list'] else {}
                
                volume_24h = float(ticker_data.get('turnover24h', 0))
                status = inst.get('status', '')
                
                if (volume_24h >= Config.MIN_24H_VOLUME and 
                    status == 'Trading' and
                    'USDT' in symbol):
                    
                    pair = TradingPair(
                        symbol=symbol,
                        base_currency=inst['baseCoin'],
                        quote_currency=inst['quoteCoin'],
                        volume_24h=volume_24h,
                        status=status
                    )
                    filtered_pairs.append(pair)
            
            self.active_pairs = filtered_pairs
            self.last_update = time.time()
            
            logging.info(f"✅ Updated pairs: {len(self.active_pairs)} active")
            return self.active_pairs
            
        except Exception as e:
            logging.error(f"Error updating pairs: {e}")
            return self.active_pairs
    
    def get_active_pairs(self) -> List[TradingPair]:
        return self.active_pairs


# ============================================================================
# GRAPH BUILDER
# ============================================================================

class GraphBuilder:
    def __init__(self):
        self.graph = nx.DiGraph()
        
    def build_graph(self, pairs: List[TradingPair], orderbooks: Dict[str, OrderBook]) -> nx.DiGraph:
        """Build directed graph"""
        self.graph.clear()
        
        for pair in pairs:
            orderbook = orderbooks.get(pair.symbol)
            if not orderbook or not orderbook.best_bid or not orderbook.best_ask:
                continue
            
            if orderbook.spread > Config.MAX_SPREAD:
                continue
            
            base = pair.base_currency
            quote = pair.quote_currency
            
            self.graph.add_node(base)
            self.graph.add_node(quote)
            
            # BUY
            buy_price = orderbook.best_ask
            buy_weight = -math.log(buy_price * (1 + Config.BYBIT_FEE + Config.SLIPPAGE_FACTOR))
            
            self.graph.add_edge(
                quote, base,
                symbol=pair.symbol,
                side=OrderSide.BUY,
                price=buy_price,
                weight=buy_weight
            )
            
            # SELL
            sell_price = orderbook.best_bid
            sell_weight = -math.log(1 / (sell_price * (1 - Config.BYBIT_FEE - Config.SLIPPAGE_FACTOR)))
            
            self.graph.add_edge(
                base, quote,
                symbol=pair.symbol,
                side=OrderSide.SELL,
                price=sell_price,
                weight=sell_weight
            )
        
        return self.graph
    
    def get_graph(self) -> nx.DiGraph:
        return self.graph


# ============================================================================
# ARBITRAGE ENGINE
# ============================================================================

class ArbitrageEngine:
    def __init__(self, graph_builder: GraphBuilder):
        self.graph_builder = graph_builder
        
    def find_cycles(self) -> List[ArbitrageCycle]:
        """Find profitable cycles"""
        cycles = []
        graph = self.graph_builder.get_graph()
        
        for start_currency in Config.START_CURRENCIES:
            if start_currency not in graph:
                continue
            
            for length in [3, 4]:
                found_cycles = self._find_cycles_from_node(graph, start_currency, length)
                cycles.extend(found_cycles)
        
        cycles.sort(key=lambda c: c.expected_profit, reverse=True)
        
        return cycles
    
    def _find_cycles_from_node(self, graph: nx.DiGraph, start: str, length: int) -> List[ArbitrageCycle]:
        """Find cycles from node"""
        cycles = []
        
        try:
            for path in nx.all_simple_paths(graph, start, start, cutoff=length):
                if len(path) == length + 1:
                    cycle = self._validate_cycle(graph, path)
                    if cycle and cycle.expected_profit >= Config.MIN_EXPECTED_PROFIT:
                        cycles.append(cycle)
        except Exception as e:
            logging.debug(f"Cycle search error: {e}")
        
        return cycles
    
    def _validate_cycle(self, graph: nx.DiGraph, path: List[str]) -> Optional[ArbitrageCycle]:
        """Validate cycle profitability"""
        try:
            pairs = []
            sides = []
            amount = Config.CYCLE_SIZE
            
            for i in range(len(path) - 1):
                from_curr = path[i]
                to_curr = path[i + 1]
                
                if not graph.has_edge(from_curr, to_curr):
                    return None
                
                edge_data = graph[from_curr][to_curr]
                symbol = edge_data['symbol']
                side = edge_data['side']
                price = edge_data['price']
                
                pairs.append(symbol)
                sides.append(side)
                
                if side == OrderSide.BUY:
                    amount = amount / price * (1 - Config.BYBIT_FEE - Config.SLIPPAGE_FACTOR)
                else:
                    amount = amount * price * (1 - Config.BYBIT_FEE - Config.SLIPPAGE_FACTOR)
            
            final_amount = amount
            profit_pct = (final_amount - Config.CYCLE_SIZE) / Config.CYCLE_SIZE
            
            if profit_pct >= Config.MIN_EXPECTED_PROFIT:
                return ArbitrageCycle(
                    path=path[:-1],
                    pairs=pairs,
                    sides=sides,
                    expected_profit=profit_pct,
                    expected_return=final_amount
                )
            
            return None
            
        except Exception as e:
            logging.debug(f"Cycle validation error: {e}")
            return None


# ============================================================================
# LIQUIDITY VALIDATOR
# ============================================================================

class LiquidityValidator:
    def __init__(self, market_data: MarketDataModule):
        self.market_data = market_data
    
    async def validate_cycle(self, cycle: ArbitrageCycle) -> Tuple[bool, float]:
        """Validate liquidity"""
        try:
            liquidity_scores = []
            amount = cycle.start_amount
            
            for i, (pair, side) in enumerate(zip(cycle.pairs, cycle.sides)):
                orderbook = await self.market_data.get_orderbook(pair)
                
                if not orderbook:
                    return False, 0.0
                
                if side == OrderSide.BUY:
                    available_liquidity = sum(p * q for p, q in orderbook.asks[:10])
                else:
                    available_liquidity = sum(q for p, q in orderbook.bids[:10])
                
                if available_liquidity < Config.MIN_DEPTH_USDT:
                    return False, 0.0
                
                liquidity_score = min(available_liquidity / (amount * 2), 1.0)
                liquidity_scores.append(liquidity_score)
                
                price = orderbook.best_ask if side == OrderSide.BUY else orderbook.best_bid
                if side == OrderSide.BUY:
                    amount = amount / price * (1 - Config.BYBIT_FEE)
                else:
                    amount = amount * price * (1 - Config.BYBIT_FEE)
            
            overall_score = min(liquidity_scores) if liquidity_scores else 0.0
            
            return overall_score > 0.5, overall_score
            
        except Exception as e:
            logging.error(f"Liquidity validation error: {e}")
            return False, 0.0


# ============================================================================
# CAPITAL CONTROLLER
# ============================================================================

class CapitalController:
    def __init__(self):
        self.slots = [
            ExecutionSlot(slot_id=i, capital=Config.CYCLE_SIZE)
            for i in range(1, Config.MAX_ACTIVE_CYCLES + 1)
        ]
        self.lock = asyncio.Lock()
    
    async def get_free_slot(self) -> Optional[ExecutionSlot]:
        async with self.lock:
            for slot in self.slots:
                if slot.status == SlotStatus.IDLE:
                    return slot
            return None
    
    async def allocate_slot(self, slot: ExecutionSlot, cycle: ArbitrageCycle) -> bool:
        async with self.lock:
            used_capital = sum(
                s.capital for s in self.slots 
                if s.status == SlotStatus.EXECUTING
            )
            
            if used_capital + Config.CYCLE_SIZE <= Config.WORKING_CAPITAL:
                slot.status = SlotStatus.EXECUTING
                slot.cycle = cycle
                slot.start_time = time.time()
                slot.current_step = 0
                slot.executed_orders = []
                return True
            
            return False
    
    async def release_slot(self, slot: ExecutionSlot, status: SlotStatus):
        async with self.lock:
            slot.status = status
            slot.cycle = None
    
    def get_active_count(self) -> int:
        return sum(1 for s in self.slots if s.status == SlotStatus.EXECUTING)
    
    def get_used_capital(self) -> float:
        return sum(
            s.capital for s in self.slots 
            if s.status == SlotStatus.EXECUTING
        )


# ============================================================================
# EXECUTION ENGINE
# ============================================================================

class ExecutionEngine:
    def __init__(self, http_client: HTTP, market_data: MarketDataModule):
        self.http = http_client
        self.market_data = market_data
    
    async def execute_cycle(self, slot: ExecutionSlot) -> bool:
        """Execute cycle"""
        cycle = slot.cycle
        if not cycle:
            return False
        
        try:
            current_amount = cycle.start_amount
            
            for i, (pair, side) in enumerate(zip(cycle.pairs, cycle.sides)):
                slot.current_step = i + 1
                
                if time.time() - slot.start_time > Config.MAX_CYCLE_TIME:
                    slot.error_message = "Timeout"
                    return False
                
                orderbook = await self.market_data.get_orderbook(pair)
                if not orderbook:
                    slot.error_message = f"No orderbook: {pair}"
                    return False
                
                if side == OrderSide.BUY:
                    price = orderbook.best_ask
                    qty = current_amount / price
                else:
                    price = orderbook.best_bid
                    qty = current_amount
                
                order_result = await self._place_order(pair, side, qty, price)
                
                if not order_result:
                    slot.error_message = f"Order failed at step {i+1}"
                    return False
                
                slot.executed_orders.append(order_result)
                
                if side == OrderSide.BUY:
                    current_amount = qty * (1 - Config.BYBIT_FEE)
                else:
                    current_amount = qty * price * (1 - Config.BYBIT_FEE)
            
            final_amount = current_amount
            slot.actual_profit = final_amount - cycle.start_amount
            
            if slot.actual_profit < -Config.CYCLE_SIZE * Config.EMERGENCY_STOP_LOSS:
                slot.error_message = "Stop loss"
                return False
            
            return True
            
        except Exception as e:
            slot.error_message = str(e)
            logging.error(f"Execution error slot {slot.slot_id}: {e}")
            return False
    
    async def _place_order(self, symbol: str, side: OrderSide, qty: float, price: float) -> Optional[Dict]:
        """Place market order"""
        try:
            order = self.http.place_order(
                category="spot",
                symbol=symbol,
                side="Buy" if side == OrderSide.BUY else "Sell",
                orderType="Market",
                qty=str(round(qty, 6)),
                timeInForce="IOC"
            )
            
            if order['retCode'] == 0:
                logging.info(f"✅ Order: {symbol} {side.value} qty={qty:.6f}")
                return order['result']
            else:
                logging.error(f"❌ Order failed: {order['retMsg']}")
                return None
                
        except Exception as e:
            logging.error(f"Order error: {e}")
            return None


# ============================================================================
# RISK MANAGER
# ============================================================================

class RiskManager:
    def __init__(self, metrics: BotMetrics):
        self.metrics = metrics
        self.bot_active = True
    
    def check_daily_loss_limit(self) -> bool:
        if self.metrics.daily_loss >= Config.MAX_CAPITAL * Config.DAILY_LOSS_LIMIT:
            logging.critical("🚨 DAILY LOSS LIMIT EXCEEDED")
            self.bot_active = False
            return False
        return True
    
    def is_bot_active(self) -> bool:
        return self.bot_active and self.check_daily_loss_limit()


# ============================================================================
# MAIN ARBITRAGE BOT
# ============================================================================

class ArbitrageBot:
    def __init__(self, api_key: str, api_secret: str):
        # Initialize HTTP client for DEMO
        self.http = HTTP(
            testnet=False,  # Not testnet!
            demo=True,      # DEMO mode
            api_key=api_key,
            api_secret=api_secret
        )
        
        logging.info(f"✅ HTTP client initialized for DEMO: {Config.REST_ENDPOINT}")
        
        # Initialize modules
        self.market_data = MarketDataModule()
        self.pair_filter = PairFilterModule(self.http)
        self.graph_builder = GraphBuilder()
        self.arbitrage_engine = ArbitrageEngine(self.graph_builder)
        self.liquidity_validator = LiquidityValidator(self.market_data)
        self.capital_controller = CapitalController()
        self.execution_engine = ExecutionEngine(self.http, self.market_data)
        
        # Metrics and risk
        self.metrics = BotMetrics()
        self.risk_manager = RiskManager(self.metrics)
        
        # Telegram
        self.notifier = TelegramNotifier()
        
        # Shutdown handling
        self.shutdown_requested = False
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)
        
        # Timers
        self.last_pair_update = 0
        self.last_metrics_report = 0
        self.last_system_metrics = 0
        
        logging.info("✅ ArbitrageBot initialized")
    
    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signals"""
        logging.info(f"🛑 Shutdown signal {signum} received")
        self.shutdown_requested = True
        self.risk_manager.bot_active = False
        self.notifier.send_shutdown("Signal received")
    
    async def start(self):
        """Start bot"""
        try:
            logging.info("🚀 Starting Arbitrage Bot...")
            self.notifier.send_startup()
            
            # Connect WebSocket
            await self.market_data.connect()
            
            # Initial pairs
            self.active_pairs = await self.pair_filter.update_pairs()
            
            # Subscribe
            symbols = [p.symbol for p in self.active_pairs]
            await self.market_data.subscribe_orderbook(symbols)
            
            await asyncio.sleep(3)
            
            # Main loop
            await self.main_loop()
            
        except Exception as e:
            error_msg = f"Fatal error: {e}"
            logging.error(error_msg, exc_info=True)
            self.notifier.send_error(error_msg)
            raise
    
    async def main_loop(self):
        """Main loop"""
        logging.info("🔄 Entering main loop...")
        
        while self.risk_manager.is_bot_active() and not self.shutdown_requested:
            try:
                # Update pairs
                if time.time() - self.last_pair_update > Config.PAIR_FILTER_INTERVAL:
                    self.active_pairs = await self.pair_filter.update_pairs()
                    symbols = [p.symbol for p in self.active_pairs]
                    await self.market_data.subscribe_orderbook(symbols)
                    self.last_pair_update = time.time()
                
                # Build graph
                orderbooks = self.market_data.get_all_orderbooks()
                self.graph_builder.build_graph(self.active_pairs, orderbooks)
                
                # Find cycles
                cycles = self.arbitrage_engine.find_cycles()
                
                # Execute
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
                        f"🚀 Started slot {slot.slot_id}: "
                        f"{' -> '.join(cycle.path)} "
                        f"(+{cycle.expected_profit*100:.3f}%)"
                    )
                
                # Update metrics
                self.metrics.active_slots = self.capital_controller.get_active_count()
                
                # Reports
                if time.time() - self.last_metrics_report > Config.METRICS_REPORT_INTERVAL:
                    self.notifier.send_status_report(self.metrics)
                    self.last_metrics_report = time.time()
                
                if time.time() - self.last_system_metrics > Config.SYSTEM_METRICS_INTERVAL:
                    self.notifier.send_system_metrics()
                    self.last_system_metrics = time.time()
                
                # Log
                if self.metrics.total_cycles % 10 == 0 and self.metrics.total_cycles > 0:
                    self._log_status()
                
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logging.error(f"Main loop error: {e}")
                await asyncio.sleep(1)
        
        # Wait for active cycles
        logging.info("⏳ Waiting for active cycles...")
        while self.capital_controller.get_active_count() > 0:
            await asyncio.sleep(1)
        
        logging.info("✅ Bot stopped gracefully")
    
    async def _execute_slot(self, slot: ExecutionSlot):
        """Execute slot"""
        start_time = time.time()
        
        try:
            success = await self.execution_engine.execute_cycle(slot)
            duration = time.time() - start_time
            
            profit = slot.actual_profit if success else -Config.CYCLE_SIZE * Config.EMERGENCY_STOP_LOSS
            self.metrics.record_cycle(success, profit, duration)
            
            # Notify
            cycle_path = ' -> '.join(slot.cycle.path) if slot.cycle else 'Unknown'
            self.notifier.send_cycle_complete(slot.slot_id, profit, duration, cycle_path)
            
            status = SlotStatus.COMPLETED if success else SlotStatus.FAILED
            await self.capital_controller.release_slot(slot, status)
            
            if success:
                logging.info(
                    f"✅ Slot {slot.slot_id} COMPLETED: "
                    f"+{profit:.4f} USDT ({profit/Config.CYCLE_SIZE*100:.2f}%) "
                    f"{duration:.2f}s"
                )
            else:
                logging.warning(f"❌ Slot {slot.slot_id} FAILED: {slot.error_message}")
                
        except Exception as e:
            logging.error(f"Slot {slot.slot_id} error: {e}")
            self.notifier.send_error(f"Slot {slot.slot_id}: {e}")
            await self.capital_controller.release_slot(slot, SlotStatus.FAILED)
    
    def _log_status(self):
        """Log status"""
        logging.info("=" * 60)
        logging.info(f"📊 Total: {self.metrics.total_cycles} | Win: {self.metrics.win_rate*100:.1f}%")
        logging.info(f"💰 Profit: {self.metrics.total_profit:.2f} USDT | Avg: {self.metrics.avg_profit_per_cycle:.4f}")
        logging.info(f"⚡ Slots: {self.metrics.active_slots}/{Config.MAX_ACTIVE_CYCLES}")
        logging.info("=" * 60)


# ============================================================================
# MAIN WITH AUTO-RESTART
# ============================================================================

async def main():
    """Main with auto-restart"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler('arbitrage_bot.log'),
            logging.StreamHandler()
        ]
    )
    
    logging.info("=" * 60)
    logging.info("🤖 ARBITRAGE BOT FOR BYBIT DEMO ACCOUNT")
    logging.info("=" * 60)
    
    # Load credentials
    api_key = os.getenv('BYBIT_API_KEY')
    api_secret = os.getenv('BYBIT_API_SECRET')
    
    if not api_key or not api_secret:
        logging.error("❌ Missing BYBIT_API_KEY or BYBIT_API_SECRET")
        return
    
    logging.info(f"✅ API Key: {api_key[:4]}...{api_key[-4:]}")
    logging.info(f"✅ Mode: DEMO Account")
    logging.info(f"✅ REST: {Config.REST_ENDPOINT}")
    logging.info(f"✅ WebSocket: {Config.WS_ENDPOINT}")
    
    # Auto-restart
    MAX_RESTARTS = 5
    restart_count = 0
    
    while restart_count < MAX_RESTARTS:
        try:
            bot = ArbitrageBot(api_key, api_secret)
            await bot.start()
            break
            
        except KeyboardInterrupt:
            logging.info("⌨️ Keyboard interrupt")
            break
            
        except Exception as e:
            restart_count += 1
            logging.error(
                f"💥 Critical error (restart {restart_count}/{MAX_RESTARTS}): {e}",
                exc_info=True
            )
            
            if restart_count < MAX_RESTARTS:
                wait_time = min(30 * restart_count, 300)
                logging.info(f"⏳ Restarting in {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                logging.critical("🚨 Max restarts reached")
                
                notifier = TelegramNotifier()
                notifier.send_error("🚨 Bot crashed after max restarts!")


if __name__ == "__main__":
    asyncio.run(main())
