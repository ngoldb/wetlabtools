from typing import List, Optional
from wetlabtools.plate import PlateRegion


class Action:
    """base class for actions"""

    def __init__(self, label: str):
        self.label = label
        self.parent: Optional["Action"] = None
        self.children: List["Action"] = []
    
    # can probably be removed (or been used)
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
    
    def parse_block(self, ctx):
        """Function to parse data and metadata from excel file"""
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement parse_block()"
        )
    
    def plot(self):
        """Function for default plotting of action data"""
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement parse_block()"
        )
    
    def __repr__(self):
        return f"{self.__class__.__name__}(label={self.label}, children={len(self.children)})"
    
    def __str__(self):
        return f"{self.__class__.__name__}, {self.label}"