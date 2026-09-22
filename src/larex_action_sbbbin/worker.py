# SPDX-License-Identifier: MIT
from __future__ import annotations

import logging
from dataclasses import dataclass
from multiprocessing.queues import Queue
from pathlib import Path
from typing import Literal

from larex_action_sbbbin.config import env


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO if env.LAREX_ACTION_DEBUG else logging.WARNING)


@dataclass
class WorkerResult:
    image: bytes | None = None
    error: str | None = None


def worker(
    in_queue: Queue[bytes | None],  # image bytes or stop signal (None)
    out_queue: Queue[WorkerResult],
    model: Path,
    device: Literal['auto', 'cpu', 'cuda'] = 'cuda',
    extension: str = '.png',
) -> None:
    import cv2
    import numpy as np
    from sbbbin import SbbBinarizer
    
    logger.info(f'Starting worker: model={model} device={device} ext={extension}')
    
    try:
        binarizer = SbbBinarizer(model, device)
    except Exception:
        logger.exception('Failed to initialize SBB Binarizer')
        raise
    
    logger.info('SBB Binarizer initialized successfully')
    
    while True:
        image = in_queue.get()
        if image is None:
            logger.info('Shutdown worker')
            break

        try:
            decoded = cv2.imdecode(
                np.frombuffer(image, np.uint8),
                cv2.IMREAD_COLOR
            )
            if decoded is None:
                raise ValueError('Could not decode image')
            
            result = binarizer.process(decoded)
            
            success, buffer = cv2.imencode(extension, result)
            if not success:
                raise ValueError('Failed to encode image')
            out_queue.put(WorkerResult(image=buffer.tobytes()))
        
        except Exception as exc:
            logger.exception('Failed to process image')
            out_queue.put(WorkerResult(error=f'{type(exc).__name__}: {exc}'))

    logger.info('Worker stopped')
