# SPDX-License-Identifier: MIT
from __future__ import annotations

import logging
from dataclasses import dataclass
from multiprocessing.connection import Connection
from pathlib import Path

from .config import env

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG if env.ACTION_DEBUG else logging.INFO)


@dataclass
class WorkerRequest:
    """ Worker inputs """
    image: bytes
        

@dataclass
class WorkerSuccess:
    """ Worker outputs on success """
    image: bytes
    

@dataclass
class WorkerFailure:
    """ Worker output on failure """
    error: str


def worker(connection: Connection, model: Path) -> None:
    """
    Process-isolated action worker.
    
    Request input with `request = connection.recv()`.
    Send output with `connection.send(obj)`.
    """
    logger.debug('Initialize worker')
    
    import cv2
    import numpy as np
    from sbbbin import SbbBinarizer

    try:
        binarizer = SbbBinarizer(model, env.SBB_DEVICE)
    except Exception as exc:
        logger.exception('Failed to initialize SBB Binarizer')
        connection.send(WorkerFailure(f'{type(exc).__name__}: {exc}'))
        return
    
    while True:
        try:
            request = connection.recv()
        except EOFError:
            logger.debug('Parent closed pipe; shutting down worker')
            break
        if request is None:
            logger.debug('Parent sent shutdown signal; shutting down worker')
            break   
        if not isinstance(request, WorkerRequest):
            logger.error(f'Received unexpected request: {type(request).__name__}')
            continue
        
        try:
            image = cv2.imdecode(
                np.frombuffer(request.image, np.uint8),
                cv2.IMREAD_COLOR
            )
            if image is None:
                raise ValueError('Could not decode image')
            
            result = binarizer.process(image)
            success, buffer = cv2.imencode(
                f'.{env.SBB_SUFFIX.split(".")[-1]}', 
                result
            )
            if not success:
                raise ValueError('Failed to encode image')

            connection.send(WorkerSuccess(buffer.tobytes()))
            
        except Exception as exc:
            logger.exception('Failed to process element')
            try:
                connection.send(WorkerFailure(f'{type(exc).__name__}: {exc}'))
            except (BrokenPipeError, EOFError, OSError):
                logger.warning('Could not send worker failure to parent')
                break
    
    connection.close()
    logger.info('Worker stopped successfully')
