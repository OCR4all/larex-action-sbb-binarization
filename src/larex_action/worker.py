# SPDX-License-Identifier: MIT
import logging
from dataclasses import dataclass
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Literal

from .config import env

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG if env.ACTION_DEBUG else logging.INFO)


@dataclass
class WorkerRequest:
    """ Inputs for the worker process """
    image: bytes


@dataclass
class WorkerResponse:
    """ Worker outputs on success """
    image: bytes


@dataclass
class WorkerStatus:
    """ Status message from the worker """
    type: Literal['loaded', 'error']
    message: str | None = None


def worker(connection: Connection, model: Path) -> None:
    """
    Process-isolated action worker.
 
    Protocol:
    1. Send exactly one `WorkerStatus` after startup: 'loaded', or 'error' followed by exit if loading failed.
    2. For every request (`request = connection.recv()`), send exactly one `WorkerResponse`
       or `WorkerStatus('error', ...)` (`connection.send(obj)`).
    3. Exit when `None` is received or the parent closes the pipe.
    """
    logger.debug('Initialize worker')
    
    try:
        import cv2
        import numpy as np
        from sbbbin import SbbBinarizer
        
        binarizer = SbbBinarizer(model, env.SBB_DEVICE)
        
        connection.send(WorkerStatus('loaded'))
    except Exception as exc:
        logger.exception('Failed to load worker')
        try:
            connection.send(WorkerStatus('error', f'{type(exc).__name__} {exc}'))
        except (BrokenPipeError, EOFError, OSError):
            logger.warning('Could not send load failure to parent')
        connection.close()
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
            
            connection.send(WorkerResponse(buffer.tobytes()))

        except Exception as exc:
            logger.exception('Failed to process element')
            try:
                connection.send(WorkerStatus('error', f'{type(exc).__name__} {exc}'))
            except (BrokenPipeError, EOFError, OSError):
                logger.warning('Could not send worker failure to parent')
                break

    connection.close()
    logger.info('Worker stopped successfully')
