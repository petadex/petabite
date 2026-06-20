"""Generic name->class registry for factory patterns."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class Registry:
    """Maps string keys to classes for config-driven construction."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._table: dict[str, type] = {}

    def register(self, key: str) -> Callable[[type[T]], type[T]]:
        """Decorator registering a class under ``key``."""

        def decorator(cls: type[T]) -> type[T]:
            if key in self._table:
                logger.warning("Overwriting %s registry key '%s'", self._name, key)
            self._table[key] = cls
            return cls

        return decorator

    def get(self, key: str) -> type:
        """Return the class registered under ``key``."""
        if key not in self._table:
            raise KeyError(
                f"Unknown {self._name} key '{key}'. Valid keys: {sorted(self._table)}"
            )
        return self._table[key]

    def keys(self) -> list[str]:
        """Return registered keys."""
        return sorted(self._table)
