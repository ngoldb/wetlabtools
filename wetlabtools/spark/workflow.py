'''Module to create protocols from Tecan experiment files'''

from typing import List, Optional, Dict, Any
from wetlabtools.spark.actions import Action, create_action


class Workflow:
    '''
    Class to store protocol information from a Tecan Spark experiment
    Orders actions according to the hierarchy from Tecan Spark
    Handles relationships of different actions
    '''
    
    def __init__(self):
        self.root_actions: List[Action] = []

    def add_action(self, action: Action) -> None:
        """Add a root-level action to the protocol."""
        self.root_actions.append(action)

    def find_all(self, action_type: type=None) -> List[Action]:
        """Find all actions of a given type in the whole protocol."""
        matches = []
        for action in self.root_actions:
            if action_type == None or isinstance(action, action_type):
                matches.append(action)
            matches.extend(action.find_descendants(action_type))
        return matches
    
    def to_tree(self) -> str:
        """
        Print the entire protocol hierarchy.
        """
        desc = f"{self}"
        for action in self.root_actions:
            desc += "\n" + action.to_tree(1)
        return desc

    def __repr__(self):
        return f"{self.__class__.__name__}(root_actions={len(self.root_actions)})"
    


def workflow_from_action_list(action_list) -> Workflow:
    """
    Build a Workflow tree from a list of (hierarchy_level, tokens).
    """

    workflow = Workflow()
    stack: list[Action] = []

    for i, (level, tokens) in enumerate(action_list):
        action = create_action(tokens)

        if level == 0:
            workflow.add_action(action)
            stack = [action]
            continue

        if level > len(stack):
            raise ValueError(
                f"Invalid hierarchy jump at index {i}: "
                f"got level {level}, but current depth is {len(stack)}"
            )

        parent = stack[level - 1]
        parent.add_child(action)

        # keep only up to the parent level, then push this action
        stack = stack[:level]
        stack.append(action)

    return workflow