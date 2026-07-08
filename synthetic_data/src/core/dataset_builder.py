"""
core/dataset_builder.py - Incremental DataFrame construction helper.

Provides a stateful builder that accumulates feature columns from each
generator and engine into a single Pandas DataFrame.

Design note:
    Rather than passing raw dictionaries between pipeline stages, the
    DatasetBuilder acts as the shared state container. Each generator
    adds its columns to the builder; the pipeline retrieves the final
    DataFrame at the end.

Usage:
    builder = DatasetBuilder(n_samples=10_000)
    builder.add_columns({"age_month": array, "gender": array})
    df = builder.to_dataframe()
"""

import pandas as pd
import numpy as np
from numpy.typing import ArrayLike

from synthetic_data.src.utils.logger import get_logger

logger = get_logger(__name__)


class DatasetBuilder:
    """
    Accumulates feature arrays into a single Pandas DataFrame.

    Attributes
    ----------
    n_samples:
        Expected number of rows. Used to validate incoming arrays.
    """

    def __init__(self, n_samples: int) -> None:
        self.n_samples: int = n_samples
        self._columns: dict[str, ArrayLike] = {}

    def add_column(self, name: str, values: ArrayLike) -> None:
        """
        Add a single feature column.

        Parameters
        ----------
        name:
            Column name.
        values:
            Array-like with exactly ``n_samples`` elements.

        Raises
        ------
        ValueError
            If the length of ``values`` does not match ``n_samples``.
        """
        arr = np.asarray(values) if not isinstance(values, np.ndarray) else values
        if len(arr) != self.n_samples:
            raise ValueError(
                f"Column '{name}' has {len(arr)} rows, expected {self.n_samples}."
            )
        if name in self._columns:
            logger.warning("Overwriting existing column: '%s'", name)
        self._columns[name] = arr

    def add_columns(self, columns: dict[str, ArrayLike]) -> None:
        """
        Add multiple feature columns at once.

        Parameters
        ----------
        columns:
            Mapping from column name to array-like values.
        """
        for name, values in columns.items():
            self.add_column(name, values)

    def update_column(self, name: str, values: ArrayLike) -> None:
        """
        Replace an existing column (used by Relationship Engine).

        Raises
        ------
        KeyError
            If the column does not already exist.
        """
        if name not in self._columns:
            raise KeyError(
                f"Cannot update column '{name}': it does not exist in the builder. "
                "Use add_column() instead."
            )
        arr = np.asarray(values) if not isinstance(values, np.ndarray) else values
        if len(arr) != self.n_samples:
            raise ValueError(
                f"Column '{name}' replacement has {len(arr)} rows, expected {self.n_samples}."
            )
        self._columns[name] = arr

    def has_column(self, name: str) -> bool:
        """Return True if a column with this name has been added."""
        return name in self._columns

    def get_column(self, name: str) -> np.ndarray:
        """
        Retrieve a column by name.

        Raises
        ------
        KeyError
            If the column does not exist.
        """
        if name not in self._columns:
            raise KeyError(f"Column '{name}' not found in DatasetBuilder.")
        return np.asarray(self._columns[name])

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert the accumulated columns to a Pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with all added columns.
        """
        df = pd.DataFrame(self._columns)
        logger.debug(
            "DatasetBuilder → DataFrame: %d rows × %d columns",
            len(df),
            len(df.columns),
        )
        return df

    @property
    def column_names(self) -> list[str]:
        """List of all column names added so far."""
        return list(self._columns.keys())

    @property
    def shape(self) -> tuple[int, int]:
        """(n_samples, n_columns) - only valid after at least one column is added."""
        return (self.n_samples, len(self._columns))

    def __repr__(self) -> str:
        return (
            f"DatasetBuilder(n_samples={self.n_samples}, "
            f"columns={self.column_names})"
        )
