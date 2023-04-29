from abc import ABC
from typing import Any, List, Mapping

class BaseModel(ABC):
    def completions(self, request: Mapping[str, Any]):
        pass