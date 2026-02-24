import datetime
from wetlabtools.spark.actions.base_action import Action
from wetlabtools.spark.actions.plate_action import PlateAction
from wetlabtools.spark.actions.kinetic_action import KineticAction

from wetlabtools.plate import PlateRegion

from wetlabtools.spark.action_registry import register_action
from wetlabtools.spark.parse import block_2_dict, df_from_plate_like_block


@register_action("fluorescence top reading")
class FluorescenceAction(Action):
    '''class for fluorescence measurements'''

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
            lambda row: row[0]=="Mode" and "Fluorescence" in row[1],
            drop_empty=False
        )
        # self.mode = ctx.peek(-1, drop_empty=True)[1]
        # ctx.advance()
        settings_block = ctx.read_until_empty_row(drop_empty=True)
        settings_dict = block_2_dict(settings_block)
        self.mode = settings_dict["Mode"]
        parent_plate = self.get_parent(PlateAction)
        wells = parent_plate.total_wells
        self.region = PlateRegion(settings_dict['Part of Plate'], wells_total=wells)

        if any(["Multiple Reads per Well" in key for key in settings_dict.keys()]):
            self.multiple_reads = True
        else:
            self.multiple_reads = False

        ctx.advance()
        meta_data_block = ctx.read_until_empty_row(drop_empty=True)
        meta_data_dict = block_2_dict(meta_data_block)
        self.start_time = datetime.datetime.strptime(meta_data_dict['Start Time'], "%Y-%m-%d %H:%M:%S")
        
        if not self.kinetic:
            self.temperature = float(meta_data_dict['Temperature [°C]'])
        
    
    def _parse_data(self, ctx):
        if self.kinetic:
            raise NotImplementedError('Parsing data from fluorescence measurements inside kinetic loop not implemented')
        elif self.multiple_reads:
            raise NotImplementedError('Parsing data from fluorescence measurements with multiple reads per well not implemented')
        else:
            _ = ctx.read_until(
                    lambda row: row[0]=='<>',
                    drop_empty=False
            )
            data_block = ctx.read_until_empty_row(drop_empty=False)
            self.data = df_from_plate_like_block(data_block, data_label='value')