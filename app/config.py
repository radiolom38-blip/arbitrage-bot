import os

class Config:
    def __init__(self):
        self.API_KEY = os.getenv('BYBIT_API_KEY')
        self.API_SECRET = os.getenv('BYBIT_API_SECRET')
        self.DEMO_MODE = os.getenv('DEMO_MODE', 'True') == 'True'
        self.MAX_CAPITAL = 333
        self.WORKING_CAPITAL = 300
        self.RESERVE = 33
        self.CYCLE_SIZE = 50
        self.MAX_ACTIVE_CYCLES = 6
        self.MIN_EXPECTED_PROFIT = 0.0015  # 0.15%
        self.DAILY_LOSS_LIMIT = 0.03  # 3%
        self.MAX_CYCLE_TIME = 5  # seconds
        self.FEE_RATE = 0.001  # Assume 0.1% fee, adjust based on Bybit docs
        self.MIN_VOLUME_24H = 1000000  # Example threshold
        self.MAX_SPREAD_PCT = 0.002  # 0.2%
        self.MIN_DEPTH_USDT = 50
        self.START_CURRENCIES = ['USDT', 'USDC', 'BTC', 'ETH']