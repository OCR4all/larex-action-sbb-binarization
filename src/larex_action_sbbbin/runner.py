# SPDX-License-Identifier: MIT
from __future__ import annotations

import asyncio
import logging
import multiprocessing
import queue
from multiprocessing.queues import Queue

from larex_actions import ActionContext

from larex_action_sbbbin.config import env
from larex_action_sbbbin.worker import WorkerResult, worker


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO if env.LAREX_ACTION_DEBUG else logging.WARNING)


def progress(index: int, total: int) -> int:
    return int((index / total) * 100)


async def _run(ctx: ActionContext) -> None:
    action_input = await ctx.pull_input()
    if not action_input.pages:
        await ctx.fail('SBB Binarization received no pages')
        return
    
    model = action_input.parameters.get('Model')
    if model is None:
        await ctx.fail('SBB Binarization received no model')
        return
    
    model_directory = env.SBB_MODEL_DIR.expanduser().resolve()
    model_path = (model_directory / model).resolve()
    
    total = len(action_input.pages)
    
    logger.info(f'Starting worker for {total} page(s)')
    
    mp_context = multiprocessing.get_context('spawn')
    in_queue: Queue[bytes | None] = mp_context.Queue()
    out_queue: Queue[WorkerResult] = mp_context.Queue()
    
    process = mp_context.Process(
        target=worker,
        name='sbb-binarizer',
        args=(in_queue, out_queue, model_path, env.SBB_DEVICE, f'.{env.SBB_SUFFIX.split(".")[-1]}')
    )
    process.start()
    
    try:
        for i, page in enumerate(action_input.pages, start=1):
            await ctx.check_cancelled()
            await ctx.heartbeat(
                progress(i - 1, total),
                f'Binarize page {i}/{total}: {page.name}',
                raise_on_cancel=True,
            )
            
            async with ctx.step(f'SBB Binarization for {page.name}'):
                logger.info(f'Binarize page {i}/{total}: {page.name}')
                results = ctx.result_builder()
                
                try:
                    image = await ctx.download_bytes(page.images[0])
                    in_queue.put(image)
                    
                    # dont wait forever on worker crash
                    while True:
                        if not process.is_alive() and out_queue.empty():
                            raise RuntimeError(f'Worker exited with code {process.exitcode}')
                        try:
                            result = await asyncio.to_thread(out_queue.get, True, 0.5)
                            break
                        except queue.Empty:
                            continue
                    
                    if result.error:
                        message = f'Binarization failed for page {i}/{total}: {result.error}'
                        logger.error(message)
                    elif result.image is None:
                        message = f'Binarization returned no image for page {i}/{total}'
                        logger.error(message)
                    else:
                        results.add_image_bytes(
                            page_id=page.id,
                            content=result.image,
                            file_name=f'{page.images[0].file_name or page.name or "page"}{env.SBB_SUFFIX}',
                            variant=env.SBB_SUFFIX[1:],
                            mime_type=f'image/{env.SBB_SUFFIX.split(".")[-1]}',
                        )
                        message = f'Binarization finished for page {i}/{total}'
                        logger.info(message)
                        
                except Exception as exc:
                    message = f'Binarization failed for page {i}/{total}: {type(exc).__name__}: {exc}'
                    logger.exception(message)
                
                await ctx.submit_page_results(
                    page_id=page.id,
                    results=results,
                    message=message
                )
            
            await ctx.heartbeat(
                progress(i, total),
                f'Finished binarization for page {i}/{total}',
                raise_on_cancel=True,
            )
        
        in_queue.put(None)  # send termination signal
        await asyncio.to_thread(process.join)
        
        if process.exitcode != 0:
            message = f'SBB Binarization worker exited with code {process.exitcode}'
            logger.error(message)
            await ctx.fail(message)
            return
        
        message = f'Successfully binarized {total} page(s)'
        logger.info(message)
        await ctx.complete(message=message)
        
    except asyncio.CancelledError:
        logger.info('SBB binarization cancelled')
        raise
    
    except Exception:
        logger.exception('Unexpected SBB binarization failure')
        await ctx.fail('SBB Binarization failed')
        
    finally:
        if process.is_alive():
            logger.info('Terminating SBB worker')
            process.terminate()

        await asyncio.to_thread(process.join)

        in_queue.close()
        out_queue.close()
        in_queue.join_thread()
        out_queue.join_thread()


async def run(ctx: ActionContext) -> None:
    async with env.RUN_SEMAPHORE:
        await _run(ctx)
