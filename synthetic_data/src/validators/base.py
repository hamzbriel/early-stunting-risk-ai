"""
validators/base.py - Abstract base class for data quality validators.

Each validator inspects the final Pandas DataFrame and reports issues.
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Any

from synthetic_data.src.core.config_loader import GeneratorConfig
from synthetic_data.src.utils.logger import get_logger

logger = get_logger(__name__)


class BaseValidator(ABC):
    """
    Abstract base class for data quality and consistency checks.
    """

    def __init__(self, config: GeneratorConfig) -> None:
        self.config = config
        self.val_config: dict[str, Any] = config.validation_rules

    @abstractmethod
    def validate(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        """
        Run validation checks on the DataFrame.

        Parameters
        ----------
        df:
            The complete generated DataFrame.

        Returns
        -------
        list[dict[str, Any]]
            A list of issue dictionaries. Each issue has:
                - 'rule': name of the validated rule
                - 'message': description of the issue
                - 'severity': 'warning' or 'error'
                - 'meta': additional diagnostic info (optional)
        """
        ...
