from wetlabtools.spark.actions.base_action import Action
from wetlabtools.spark.action_registry import register_action


@register_action("fluorescence top reading")
class FluorescenceAction(Action):
    '''class for fluorescence measurements'''

    def __init__(self, label: str):
        super().__init__(label)