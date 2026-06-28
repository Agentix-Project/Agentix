"""Compatibility wrapper for Miles agentic tool-call generate bridge."""

from miles._upstream_loader import export_public, load_upstream_module

_upstream = load_upstream_module(__name__, __file__)
if _upstream is not None:
    __all__ = export_public(_upstream, globals())
else:
    from tito_gateway.vendor.miles_compat.rollout.generate_hub.agentic_tool_call import *  # noqa: F401,F403
