"""Runtime no-op compatibility for copied Miles CPU CI markers."""


def register_cpu_ci(
    est_time: float,
    suite: str,
    *,
    labels: list[str] | None = None,
    nightly: bool = False,
    disabled: str | None = None,
):
    return None
