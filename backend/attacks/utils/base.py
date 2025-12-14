"""
Base classes for attack parameter specifications.

Import these in each attack file to define ATTACK_SPEC.
"""

from typing import List, Any, Literal
from dataclasses import dataclass, field


@dataclass
class ParamSpec:
    """Specification for a single attack parameter."""
    name: str
    type: Literal["float", "int", "bool"]
    default: Any
    min: float = None
    max: float = None
    step: float = None
    description: str = ""


@dataclass
class AttackSpec:
    """Specification for an attack method."""
    id: str
    name: str
    description: str
    category: Literal["universal", "cnn", "vit"]
    params: List[ParamSpec] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "params": [
                {k: v for k, v in {
                    "name": p.name,
                    "type": p.type,
                    "default": p.default,
                    "min": p.min,
                    "max": p.max,
                    "step": p.step,
                    "description": p.description
                }.items() if v is not None}
                for p in self.params
            ]
        }
