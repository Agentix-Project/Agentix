"""Agentix's native TITO engine — token-in token-out pretokenization, session
trajectory, and mismatch-audit logic.

Model-agnostic core lives here; per-model behavior is a small amount of data
(a fixed chat template) plus a couple of constants and an optional boundary
fixup. See `pretokenize.TITOTokenizer` for the algorithm.
"""
