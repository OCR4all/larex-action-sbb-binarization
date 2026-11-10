# SPDX-License-Identifier: MIT
from __future__ import annotations

import asyncio
from typing import Literal

from pydantic import PositiveInt, computed_field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    LAREX_PROCESSOR_ID: str
    LAREX_DISPATCH_HMAC_SECRET: str
    LAREX_ALLOWED_CALLBACK_ORIGINS: list[str]
    LAREX_ROUTE_PREFIXES: list[str]
    LAREX_SDK_TRANSPORT_DEBUG: bool = False
    
    ACTION_DEBUG: bool = False
    ACTION_MAX_CONCURRENT_RUNS: PositiveInt = 1
    ACTION_PAGE_TIMEOUT: PositiveInt | None = None
    
    SBB_SUFFIX: str = '.sbb.bin.png'
    SBB_DEVICE: Literal['auto', 'cpu', 'cuda'] = 'cuda'
    
    @computed_field
    @property
    def RUN_SEMAPHORE(self) -> asyncio.Semaphore:
        return asyncio.Semaphore(self.ACTION_MAX_CONCURRENT_RUNS)

env = Settings()
