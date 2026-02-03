"""
Module for actions in Tecan Spark experiments
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from wetlabtools.spark.utilities import PlateRegion

ACTION_REGISTRY = {}

def register_action(name: str):
    def decorator(cls):
        ACTION_REGISTRY[name.lower()] = cls
        return cls
    return decorator


class Action:
    """base class for actions"""

    def __init__(self, label: str):
        self.label = label
        self.parent: Optional["Action"] = None
        self.children: List["Action"] = []
    
    def add_plate_area(self, region_str: str, wells_total: int):
        self.plate_area = PlateRegion(region_str, wells_total)
    
    def add_child(self, child: "Action") -> None:
        """Attach a child action and set parent link."""
        child.parent = self
        self.children.append(child)

    def get_parent(self, action_type: type=None) -> Optional["Action"]:
        """
        Traverse upward and return the closest parent of a given type.
        If action_type is None, return the immediate parent.
        """
        current = self.parent
        while current:
            if action_type is None or isinstance(current, action_type):
                return current
            current = current.parent
        return None
    
    def find_descendants(self, action_type: type=None) -> List["Action"]:
        """Recursively find all descendants of a given type."""
        matches = []
        for child in self.children:
            if action_type is None or isinstance(child, action_type):
                matches.append(child)
            matches.extend(child.find_descendants(action_type))
        return matches
    
    def to_tree(self, level: int = 0) -> str:
        """
        Return a tree representation of this action
        and all its descendants.
        """
        indent = "    " * level
        desc = f"{indent}- {self}"
        for child in self.children:
            desc += "\n" + child.to_tree(level + 1)
        return desc
    
    def __repr__(self):
        return f"{self.__class__.__name__}(label={self.label}, children={len(self.children)})"
    
    def __str__(self):
        return f"{self.__class__.__name__}, {self.label}"


@register_action("plate")
class PlateAction(Action):
    """Class for plate strip"""

    def __init__(self):
        super().__init__(label='')


@register_action("kinetic")
class KineticAction(Action):
    '''class for kinetic actions'''

    def __init__(self):
        super().__init__(label='KineticAction')

@register_action("absorbance")
class AbsorbanceAction(Action):
    '''action class for absorbance measurements'''

    def __init__(self, label: str):
        super().__init__(label)


@register_action("luminescence")
class LuminescenceAction(Action):
    '''class for luminescence measurements'''

    def __init__(self, label: str):
        super().__init__(label)


@register_action("fluorescence top reading")
class FluorescenceAction(Action):
    '''class for fluorescence measurements'''

    def __init__(self, label: str):
        super().__init__(label)


def create_action(tokens: list[str]):
    key = tokens[0].strip().lower()
    args = tokens[1:]

    if key not in ACTION_REGISTRY:
        valid = ", ".join(ACTION_REGISTRY.keys())
        raise ValueError(f"Unknown action '{key}'. Valid actions: {valid}")
    
    cls = ACTION_REGISTRY.get(key)
    try:
        return cls(*args)
    except TypeError as e:
        raise TypeError(
            f"Wrong arguments for {cls.__name__}: got {args}"
        ) from e

