import asyncio
import logging
from config import Config
from enum import Enum

logger = logging.getLogger(__name__)

class SlotState(Enum):
    IDLE = 1
    EXECUTING = 2
    COMPLETED = 3
    FAILED = 4

class CapitalSlot:
    def __init__(self, id, size):
        self.id = id
        self.size = size
        self.state = SlotState.IDLE
        self.lock = asyncio.Lock()

    async def acquire(self):
        await self.lock.acquire()
        self.state = SlotState.EXECUTING

    def release(self, success=True):
        self.state = SlotState.COMPLETED if success else SlotState.FAILED
        self.lock.release()

class CapitalController:
    def __init__(self, config: Config):
        self.config = config
        self.slots = [CapitalSlot(i+1, config.CYCLE_SIZE) for i in range(config.MAX_ACTIVE_CYCLES)]
        self.used_capital = 0
        self.daily_loss = 0

    def get_free_slot(self):
        for slot in self.slots:
            if slot.state == SlotState.IDLE and self.used_capital + self.config.CYCLE_SIZE <= self.config.WORKING_CAPITAL:
                self.used_capital += self.config.CYCLE_SIZE
                return slot
        return None

    def release_slot(self, slot, profit):
        self.used_capital -= self.config.CYCLE_SIZE
        if profit < 0:
            self.daily_loss += abs(profit)
        slot.release(profit > 0)