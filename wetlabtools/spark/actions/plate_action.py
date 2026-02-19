import pandas as pd
from wetlabtools.spark.actions.base_action import Action
from wetlabtools.spark.action_registry import register_action

from wetlabtools.plate import PlateRegion
from wetlabtools.spark.parse import block_2_dict, BlockMismatchError

WELL_FORMATS = {
    6:  (2, 3),
    12: (3, 4),
    24: (4, 6),
    48: (6, 8),
    96: (8, 12),
    384: (16, 24),
    1536: (32, 48)
}

@register_action("plate")
class PlateAction(Action):
    """Class for plate strip"""

    def __init__(self, label: str=""):
        super().__init__(label=label)
    
    def parse_block(self, ctx):

        _ = ctx.read_until(
            lambda row: "Name" in row,
            drop_empty=False
        )
        if "Plate" not in ctx.peek(1, drop_empty=False)[0]:
            raise BlockMismatchError(self.__class__.__name__, ctx.cursor)

        block = ctx.read_until_empty_row(drop_empty=True)
        block_dict = block_2_dict(block)
        self.label = block_dict["Name"]
        self.total_wells = int(self.label[3:-2])
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