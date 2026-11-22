"""Optional Weights & Biases wrapper (docs/architecture.md §4.16).

All scripts run through :class:`Tracker` so they work without wandb installed,
configured, or logged in. wandb is imported lazily inside :func:`init_tracker`
only, and every wandb failure degrades to a silent no-op — tracking must never
break a training run.
"""

from __future__ import annotations

from typing import Any


class Tracker:
    """Experiment tracker that no-ops unless backed by a live wandb run.

    Create instances via :func:`init_tracker`; constructing directly yields a
    no-op tracker. No method ever raises because of wandb problems.
    """

    def __init__(self, run: Any | None = None, config: dict[str, Any] | None = None) -> None:
        self._run = run
        self._config: dict[str, Any] = dict(config or {})

    @property
    def config(self) -> dict[str, Any]:
        """The run configuration passed at initialization."""
        return self._config

    def log(self, data: dict[str, Any], step: int | None = None) -> None:
        """Log a metrics dict to wandb; silent no-op when disabled or on failure.

        Parameters
        ----------
        data : dict
            Metric name to value mapping.
        step : int or None, optional
            Explicit step index; wandb auto-increments when omitted.
        """
        if self._run is None:
            return
        try:
            self._run.log(data, step=step)
        except Exception:
            pass

    def finish(self) -> None:
        """Finalize the wandb run (if any) and become a permanent no-op."""
        if self._run is None:
            return
        try:
            self._run.finish()
        except Exception:
            pass
        self._run = None


def init_tracker(project: str, config: dict[str, Any], enabled: bool) -> Tracker:
    """Create a :class:`Tracker`, attempting a wandb run only when enabled.

    Parameters
    ----------
    project : str
        wandb project name.
    config : dict
        Run configuration (hyperparameters etc.), also exposed via
        :attr:`Tracker.config`.
    enabled : bool
        When False, wandb is not even imported.

    Returns
    -------
    Tracker
        Backed by a wandb run if `enabled` and ``wandb.init`` succeeded;
        otherwise a silent no-op tracker.
    """
    if not enabled:
        return Tracker(config=config)
    try:
        import wandb

        run = wandb.init(project=project, config=config)
        return Tracker(run=run, config=config)
    except Exception:
        return Tracker(config=config)
