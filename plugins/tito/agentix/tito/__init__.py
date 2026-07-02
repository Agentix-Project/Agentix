"""Agentix TITO plugin — token-in-token-out session-recording gateway.

A native reimplementation of the TITO token-alignment engine (see
`agentix.tito.engine`); no vendored training-framework code and no sglang
dependency.
"""

from .config import TITOGatewayConfig
from .discovery import discover_backend_url
from .gateway import TITOGateway
from .server import SessionServer
from .tokenizer import TITOTokenizerType, get_tito_tokenizer

__all__ = [
    "TITOGateway",
    "TITOGatewayConfig",
    "SessionServer",
    "TITOTokenizerType",
    "discover_backend_url",
    "get_tito_tokenizer",
]
