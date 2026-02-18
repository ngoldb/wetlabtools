"""
Module for actions in Tecan Spark experiments
"""

import datetime
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from wetlabtools.plate import PlateRegion
from wetlabtools.spark.parse import block_2_dict, BlockMismatchError

ACTION_REGISTRY = {}
WELL_FORMATS = {
    6:  (2, 3),
    12: (3, 4),
    24: (4, 6),
    48: (6, 8),
    96: (8, 12),
    384: (16, 24),
    1536: (32, 48)
}

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
    
    def parse_block(self, ctx):
        """Function to parse data and metadata from excel file"""
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement parse_block()"
        )
    
    def __repr__(self):
        return f"{self.__class__.__name__}(label={self.label}, children={len(self.children)})"
    
    def __str__(self):
        return f"{self.__class__.__name__}, {self.label}"


@register_action("plate")
class PlateAction(Action):
    """Class for plate strip"""

    def __init__(self):
        super().__init__(label='')
    
    def parse_block(self, ctx):

        _ = ctx.read_until(
            lambda row: "Name" in row,
            drop_empty=False
        )
        if "Plate" not in ctx.peek(1, drop_empty=False)[0]:
            raise BlockMismatchError(self.__class__.__name__, ctx.cursor)

        block = ctx.read_until_empty_row(drop_empty=True)
        block_dict = block_2_dict(block)
        self.name = block_dict["Name"]
        self.total_wells = int(self.name[3:-2])
        self.rows, self.cols = WELL_FORMATS[self.total_wells]
        self.region = PlateRegion(region_str=block_dict["Plate area"], wells_total=self.total_wells)

        # parsing this weird plate layout thing
        if ctx.peek(1, drop_empty=False)[0] == '<>':
            ctx.advance(1)
            block = ctx.read_until_empty_row(drop_empty=False)
            self.layout = pd.DataFrame(
                [r[1:self.cols+1] for r in block[1:]],
                columns=block[0][1:self.cols+1],
                index=[r[0] for r in block[1:]]
            )
        else:
            self.layout = None


@register_action("kinetic")
class KineticAction(Action):
    '''class for kinetic actions'''

    def __init__(self):
        super().__init__(label='Kinetic Loop')

    def parse_block(self, ctx):
        _ = ctx.read_until(
            lambda row: row[0] == "Mode" and row[1] == "Kinetic",
            drop_empty=False
        )
        block = ctx.read_until_empty_row(drop_empty=True)

        if 'cycles' in block[1][0]:
            self.mode = "cycles"
        else:
            self.mode = "duration"

        if self.mode == "cycles":
            self.cycles = int(block[1][1])
        else:
            t = datetime.datetime.strptime(block[1][1], "%H:%M:%S")
            self.duration = self.interval_time = datetime.timedelta(
                hours=t.hour,
                minutes=t.minute,
                seconds=t.second
            )

        interval_time = block[2][1]
        if interval_time == 'Not defined':
            self.interval_time = None
        else:
            t = datetime.datetime.strptime(interval_time, "%H:%M:%S")
            self.interval_time = datetime.timedelta(
                hours=t.hour,
                minutes=t.minute,
                seconds=t.second
            )

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


@register_action("temperature")
class TemperatureAction(Action):
    '''class for temperature control'''

    def __init__(self):
        super().__init__(label="Temperature Control")
    
    def parse_block(self, ctx):
        _ = ctx.read_until(
            lambda row: "Start Time" in row[0],
            drop_empty=False
        )
        if "Temperature" not in ctx.peek(1, drop_empty=False):
            raise BlockMismatchError(self.__class__.__name__, ctx.cursor)
        block = ctx.read_until_empty_row(drop_empty=True)
        block_dict = block_2_dict(block)

        self.start_time = datetime.datetime.strptime(block_dict['Start Time'], "%Y-%m-%d %H:%M:%S")
        self.control = block_dict['Temperature control']
        self.target_temp = int(block_dict["Target temperature"][0])
        self.temp_unit = block_dict["Target temperature"][1]


@register_action("shaking")
class ShakingAction(Action):
    '''class for temperature control'''

    def __init__(self):
        super().__init__(label="Shaking Control")


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

