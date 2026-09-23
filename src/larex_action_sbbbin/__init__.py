# SPDX-License-Identifier: MIT
from __future__ import annotations

import logging
import warnings
from pathlib import Path

from fastapi.responses import JSONResponse
from larex_actions import ParameterChoice
from larex_actions.fastapi import create_larex_action_app

from larex_action_sbbbin.config import env
from larex_action_sbbbin.runner import run


def configure_action_logging() -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format='%(levelname)s:\t  %(name)s - %(message)s',
    )
    
    logger = logging.getLogger('larex_actions.transport')
    logger.setLevel(logging.DEBUG if env.LAREX_SDK_TRANSPORT_LOGGING else logging.WARNING)
    
    warnings.filterwarnings(
        'ignore',
        message=r'Using a non-tuple sequence for multidimensional indexing is deprecated.*',
        category=UserWarning,
        module=r'onnx2torch\.node_converters\.slice',
    )


def discover_sbb_models(models_directory: Path = env.SBB_MODEL_DIR) -> list[ParameterChoice]:
    return [
        ParameterChoice(value=path.name, label=path.stem)
        for path in models_directory.expanduser().absolute().glob('*.pth')
    ]

configure_action_logging()

app = create_larex_action_app(
    processor_id=env.LAREX_PROCESSOR_ID,
    dispatch_secret=env.LAREX_DISPATCH_HMAC_SECRET,
    allowed_callback_origins=env.LAREX_ALLOWED_CALLBACK_ORIGINS,
    route_prefixes=env.LAREX_ACTION_ROUTE_PREFIXES,
    handler=run,
    parameter_value_providers={'models': discover_sbb_models}
)


@app.get('/ready')
async def readiness():
    if env.RUN_SEMAPHORE.locked():
        return JSONResponse({'status': 'busy'}, status_code=503)
    return {'status': 'ready', 'capacity': env.SBB_MAX_CONCURRENT_RUNS}
