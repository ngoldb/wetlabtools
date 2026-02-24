import datetime
import pandas as pd

from wetlabtools.plate import PlateRegion
from wetlabtools.spark.parse import block_2_dict
from wetlabtools.spark.action_registry import register_action

from .base_action import Action
from .plate_action import PlateAction
from .kinetic_action import KineticAction


@register_action("luminescence")
class LuminescenceAction(Action):
    '''class for luminescence measurements'''

    def __init__(self, label: str):
        super().__init__(label)

    def parse_block(self, ctx):
        self._parse_settings(ctx)
        self._parse_data(ctx)
        # return super().parse_block(ctx)


    def _parse_settings(self, ctx):

        # determine kinetic mode
        if self.get_parent(KineticAction):
            self.kinetic = True
        else:
            self.kinetic = False
    
        _ = ctx.read_until(
            lambda row: row[0]=="Mode" and row[1]=="Luminescence",
            drop_empty=False
        )
        settings_block = ctx.read_until_empty_row(drop_empty=True)
        settings_dict = block_2_dict(settings_block)
        self.attenuation = settings_dict['Attenuation']
        self.settle_time = settings_dict['Settle time [ms]']
        self.integration_time = settings_dict['Integration time [ms]']
        self.unit = settings_dict['Output']

        parent_plate = self.get_parent(PlateAction)
        wells = parent_plate.total_wells
        self.region = PlateRegion(settings_dict['Part of Plate'], wells_total=wells)

        ctx.advance()
        meta_data_block = ctx.read_until_empty_row(drop_empty=True)
        meta_data_dict = block_2_dict(meta_data_block)
        self.start_time = datetime.datetime.strptime(meta_data_dict['Start Time'], "%Y-%m-%d %H:%M:%S")
    
        if any(["Multiple Reads per Well" in key for key in settings_dict.keys()]):
            self.multiple_reads = True
        else:
            self.multiple_reads = False

    
    def _parse_data(self, ctx):
        if self.multiple_reads:
            raise NotImplementedError("Parsing data from fluorescence measurements with multiple reads per well not implemented")
        
        if self.kinetic:
            _ = ctx.read_until(
                lambda row: self.label in row[0],
                drop_empty=False
            )
            ctx.advance()
            data_block = ctx.read_until_empty_row()
            data_df = pd.DataFrame(
                [r[1:] for r in data_block[1:]],
                columns=data_block[0][1:],
                index=[r[0] for r in data_block[1:]]
            )
            data_df.index.name = data_block[0][0]
            self.data = data_df.reset_index().melt(id_vars=["Cycle Nr.", "Time [s]", "Temp. [°C]"], var_name="Well")

        else:
            pass