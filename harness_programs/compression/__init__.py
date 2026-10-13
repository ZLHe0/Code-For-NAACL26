"""Stage 1: optimize the compression program H0 (Algorithm 1, lines 1-3).

``python -m harness_programs.compression --help`` runs a whole campaign; :class:`Campaign` is the
same thing as a library call. Each operator is importable on its own: :mod:`.rollout`,
:mod:`.propose` (with :mod:`.refine` and :mod:`.measure`), and :mod:`.gate`.
"""

from .campaign import PHASES, Campaign, CampaignConfig
from .episodes import OptEpisode, Sizing, Target, by_task, load_episodes
from .gate import NAIVE_NAME

__all__ = ["NAIVE_NAME", "PHASES", "Campaign", "CampaignConfig", "OptEpisode", "Sizing", "Target", "by_task",
           "load_episodes"]
