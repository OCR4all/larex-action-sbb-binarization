# SPDX-License-Identifier: MIT
from __future__ import annotations

import asyncio
import logging
import multiprocessing
from multiprocessing.connection import Connection
from multiprocessing.context import SpawnProcess
from pathlib import Path

from larex_actions import ActionContext

from .config import env
from .worker import WorkerFailure, WorkerRequest, WorkerSuccess, worker

logger = logging.getLogger(__name__)


def _calculate_progress(page: int, total: int) -> int:
    return int((page / total) * 100)


async def _wait_for_worker(
    ctx: ActionContext,
    process: SpawnProcess,
    connection: Connection
) -> WorkerSuccess | WorkerFailure:
    """
    Wait for the worker to return a result for the current page.
    Args:
        ctx: The ActionContext provided by the SDK.
        process: The worker process to monitor.
        connection: The pipe connection used to receive the worker's result.
    Raises:
        TimeoutError: If the page timeout is reached.
        RuntimeError: If the worker exits unexpectedly before returning a result.
    Returns:
        The worker's result, either a WorkerSuccess or WorkerFailure instance.
    """
    loop = asyncio.get_running_loop()
    if env.ACTION_PAGE_TIMEOUT is not None:
        deadline = loop.time() + env.ACTION_PAGE_TIMEOUT
    else:
        deadline = None

    while True:
        await ctx.check_cancelled()

        if deadline is not None:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise TimeoutError(f'Worker timed out after {env.ACTION_PAGE_TIMEOUT} seconds')
            poll_timeout = min(0.1, remaining)
        else:
            poll_timeout = 0.1
        
        if connection.poll(poll_timeout):
            return await asyncio.to_thread(connection.recv)
        if not process.is_alive():
            raise RuntimeError(f'Worker exited unexpectedly with code {process.exitcode}')


async def _stop_worker(process: SpawnProcess, connection: Connection) -> None:
    """
    Stop the worker gracefully. If it does not exit within the grace period, terminate it.
    Args:
        process: The worker process to stop.
        connection: The pipe connection used to send data to the worker.
    """
    if not process.is_alive():
        await asyncio.to_thread(process.join)
        return
    
    try:
        connection.send(None)
    except (BrokenPipeError, EOFError, OSError):
        pass  # The worker is already gone or the pipe is broken
    await asyncio.to_thread(process.join, 10)  # Give the worker a chance to shut down cleanly

    if process.is_alive():
        logger.warning('Worker did not stop gracefully; terminating process')
        process.terminate()
        await asyncio.to_thread(process.join)


async def _run(ctx: ActionContext) -> None: 
    action_input = await ctx.pull_input()
    if not action_input.pages:
        await ctx.fail('Action input is empty')
        
    await ctx.heartbeat(status_message='Initialize action', raise_on_cancel=True)
    
    total = len(action_input.pages)
    succeeded = 0
    logger.info(f'Starting worker for {total} pages(s)')
    
    model = action_input.parameters.get('Model')
    if model is None:
        await ctx.fail('No model set')
    else:
        model = Path(f'/models/{model}')
    
    context = multiprocessing.get_context('spawn')
    connection, child_connection = context.Pipe()
    
    process = context.Process(target=worker, args=(child_connection, model, ))

    process.start()
    child_connection.close()
    
    try:
        for i, page in enumerate(action_input.pages, start=1):
            await ctx.check_cancelled()
            await ctx.heartbeat(
                progress_percent=_calculate_progress(i - 1, total),
                status_message=f'{page.name} ({i}/{total})',
                raise_on_cancel=True
            )
            
            logger.debug(f'[{i}/{total}] Processing page {page.name}')
            results = ctx.result_builder()
            
            try:
                image = await ctx.download_bytes(page.images[0])
                connection.send(WorkerRequest(image))
                
                result = await _wait_for_worker(ctx, process, connection)
                if isinstance(result, WorkerFailure):
                    message = f'[{i}/{total}] Action failed for {page.name}: {result.error}'
                    status = 'FAILED'
                    logger.error(message)
                elif isinstance(result, WorkerSuccess):
                    results.add_image_bytes(
                        page_id=page.id,
                        content=result.image,
                        file_name=f'{page.images[0].file_name or page.name or "page"}{env.SBB_SUFFIX}',
                        variant='sbb-bin',
                        mime_type=f'image/{env.SBB_SUFFIX.split(".")[-1]}',
                    )
                    
                    message = f'[{i}/{total}] Action finished for {page.name}'
                    status = 'SUCCEEDED'
                    logger.debug(message)
                    succeeded += 1
                else:
                    raise RuntimeError(  # noqa: TRY004
                        f'Worker returned an unknown result type: {type(result).__name__}'
                    )
                
            except asyncio.CancelledError:
                raise
            except TimeoutError:
                message = f'[{i}/{total}] Action timed out for {page.name}'
                status = 'FAILED'
                logger.exception(message)
            except Exception as exc:
                message = f'[{i}/{total}] Action failed for {page.name}: {type(exc).__name__} {exc}'
                status = 'FAILED'
                logger.exception(message)
            
            await ctx.submit_page_results(page.id, results, message, page_status=status, reason=message)
            await ctx.heartbeat(progress_percent=_calculate_progress(i, total), raise_on_cancel=True)
        
        await _stop_worker(process, connection)
        if process.exitcode != 0:
            message = f'Worker exited with code {process.exitcode}'
            logger.error(message)
            await ctx.fail(message)
            return
        
        message = f'Successfully processed {succeeded}/{total} element(s)'
        logger.info(message)
        await ctx.complete(message=message)
        
    except asyncio.CancelledError:
        logger.info('Action cancelled')
        raise
    
    except Exception:
        logger.exception('Unexpected action failure')
        await ctx.fail('Action failed')
        
    finally:
        if process.is_alive():
            logger.debug('Terminating action worker')
            process.terminate()
            await asyncio.to_thread(process.join)
        connection.close()


async def runner(ctx: ActionContext) -> None:
    async with env.RUN_SEMAPHORE:
        await _run(ctx)

