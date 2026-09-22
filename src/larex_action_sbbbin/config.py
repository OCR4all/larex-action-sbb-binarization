# SPDX-License-Identifier: MIT
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Literal

from pydantic import PositiveInt, computed_field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    LAREX_PROCESSOR_ID: str = 'sbb-binarizaton'
    LAREX_DISPATCH_HMAC_SECRET: str
    LAREX_ALLOWED_CALLBACK_ORIGINS: list[str] = ['http://app:8080']
    LAREX_ACTION_ROUTE_PREFIXES: list[str] = ['/sbb-binarizaton']
    LAREX_ACTION_DEBUG: bool = False
    LAREX_SDK_TRANSPORT_LOGGING: bool = False
    
    SBB_SUFFIX: str = '.sbb.bin.png'
    SBB_DEVICE: Literal['auto', 'cpu', 'cuda'] = 'cuda'
    SBB_MODEL_DIR: Path
    SBB_MAX_CONCURRENT_RUNS: PositiveInt = 1
    
    @computed_field
    @property
    def RUN_SEMAPHORE(self) -> asyncio.Semaphore:
        return asyncio.Semaphore(self.SBB_MAX_CONCURRENT_RUNS)


env = Settings()
