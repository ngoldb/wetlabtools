import datetime
from wetlabtools.spark.actions.base_action import Action
from wetlabtools.spark.action_registry import register_action
from wetlabtools.spark.parse import block_2_dict, BlockMismatchError


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
        if "Temperature" not in ''.join(ctx.peek(1, drop_empty=False)):
            raise BlockMismatchError(self.__class__.__name__, ctx.cursor)
        block = ctx.read_until_empty_row(drop_empty=True)
        block_dict = block_2_dict(block)

        self.start_time = datetime.datetime.strptime(block_dict['Start Time'], "%Y-%m-%d %H:%M:%S")
        self.control = block_dict['Temperature control']
        self.target_temp = float(block_dict["Target temperature"][0])
        self.temp_unit = block_dict["Target temperature"][1]


@register_action("shaking")
class ShakingAction(Action):
    '''class for temperature control'''

    def __init__(self):
        super().__init__(label="Shaking Control")

    
    def parse_block(self, ctx):
        _ = ctx.read_until(
            lambda row: "Start Time" in row[0],
            drop_empty=False
        )
        if "Shaking" not in ''.join(ctx.peek(1, drop_empty=False)):
            raise BlockMismatchError(self.__class__.__name__, ctx.cursor)
        
        block = ctx.read_until_empty_row(drop_empty=True)
        block_dict = block_2_dict(block)
        self.start_time = datetime.datetime.strptime(block_dict['Start Time'], "%Y-%m-%d %H:%M:%S")
        self.mode = next((s for s in ["Linear", "Orbital", "Double orbital"] 
                    if any(s in key for key in block_dict)), None)
        self.settings = block_dict
        