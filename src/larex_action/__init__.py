# SPDX-License-Identifier: MIT
import logging
import warnings
from pathlib import Path

from larex_actions import ParameterChoice
from larex_actions.fastapi import create_larex_action_app

from .config import env
from .runner import runner


def configure_logging() -> None:
    logging.basicConfig(level=logging.WARNING, format='%(asctime)s | %(levelname)s | %(name)s | %(message)s')
    logging.getLogger('larex_actions.transport').setLevel(
        level=logging.DEBUG if env.LAREX_SDK_TRANSPORT_DEBUG else logging.INFO
    )
    logging.getLogger(__name__).setLevel(
        level=logging.DEBUG if env.ACTION_DEBUG else logging.INFO
    )
    warnings.filterwarnings(
        'ignore',
        message='Using a non-tuple sequence for multidimensional indexing is deprecated',
        category=UserWarning,
        module='onnx2torch',
    )
    

def discover_sbb_models() -> list[ParameterChoice]:
    return [ParameterChoice(value=path.name, label=path.stem) for path in Path('/models').glob('*.pth')]
        

configure_logging()

app = create_larex_action_app(
    handler=runner,
    processor_id=env.LAREX_PROCESSOR_ID,
    dispatch_secret=env.LAREX_DISPATCH_HMAC_SECRET,
    route_prefixes=env.LAREX_ROUTE_PREFIXES,
    allowed_callback_origins=env.LAREX_ALLOWED_CALLBACK_ORIGINS,
    parameter_value_providers={'models': discover_sbb_models}
)
