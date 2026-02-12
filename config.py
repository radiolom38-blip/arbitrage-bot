import os

class Config:
    API_KEY = os.getenv("BYBIT_API_KEY")
    API_SECRET = os.getenv("BYBIT_API_SECRET")
    DEMO_MODE = True  # фиксировано для demo

    MAX_CAPITAL = 333
    WORKING_CAPITAL = 300
    CYCLE_SIZE = 50
    MAX_ACTIVE_CYCLES = 6
    MIN_EXPECTED_PROFIT = 0.0015  # 0.15%
    FEE_RATE = 0.001
    MIN_VOLUME_24H = 1_000_000
    # другие параметры...