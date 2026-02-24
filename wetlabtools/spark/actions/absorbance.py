import datetime
import pandas as pd
import numpy as np

from wetlabtools.plate import PlateRegion
from wetlabtools.spark.action_registry import register_action
from wetlabtools.spark.parse import block_2_dict, df_from_plate_like_block

from .base_action import Action
from .plate_action import PlateAction
from .kinetic_action import KineticAction

@register_action("absorbance")
class AbsorbanceAction(Action):
    '''action class for absorbance measurements'''

    def __init__(self, label: str):
        super().__init__(label)

    def parse_block(self, ctx):
        self._parse_settings(ctx)
        self._parse_data(ctx)
    
    def _parse_settings(self, ctx):

        # determine kinetic mode
        if self.get_parent(KineticAction):
            self.kinetic = True
        else:
            self.kinetic = False

        _ = ctx.read_until(
            lambda row: row[0]=="Name" and row[1]==self.label,
            drop_empty=False
        )
        ctx.advance()
        settings_block = ctx.read_until_empty_row(drop_empty=True)
        settings_dict = block_2_dict(settings_block)
        ctx.advance()
        meta_data_block = ctx.read_until_empty_row(drop_empty=True)
        meta_data_dict = block_2_dict(meta_data_block)

        if "Wavelength step size [nm]" in settings_dict.keys():
            self.scan = True
            self.start = int(settings_dict['Wavelength start [nm]'])
            self.end = int(settings_dict['Wavelength end [nm]'])
            self.step = int(settings_dict['Wavelength step size [nm]'])
        else:
            self.scan = False
            self.wavelength = int(settings_dict['Measurement wavelength [nm]'])
            if "Reference wavelength [nm]" in settings_dict.keys():
                self.reference = int(settings_dict['Reference wavelength [nm]'])
            else:
                self.reference = None

        self.settle_time = int(settings_dict["Settle time [ms]"])
        self.flashes = int(settings_dict["Number of flashes"])

        # find parent plate
        parent_plate = self.get_parent(PlateAction)
        wells = parent_plate.total_wells
        self.region = PlateRegion(settings_dict['Part of Plate'], wells_total=wells)

        self.temperature = float(meta_data_dict['Temperature [°C]'])
        self.start_time = datetime.datetime.strptime(meta_data_dict['Start Time'], "%Y-%m-%d %H:%M:%S")
    
    
    def _parse_data(self, ctx):
        if self.kinetic:
            raise NotImplementedError("Parsing data from absorbance measurements inside kinetic loop not implemented")
        
        if self.scan:
            _ = ctx.read_until(
                lambda row: "Wavel." in row[0],
                drop_empty=False
            )
            data_block = ctx.read_until_empty_row(drop_empty=False)
            df = pd.DataFrame(
                [r[1:] for r in data_block[1:]],
                columns=data_block[0][1:],
                index=[r[0] for r in data_block[1:]]
            )
            df.index.name = "Wavelength"

            df_long = df.reset_index().melt(
                id_vars="Wavelength",
                var_name="Well",
                value_name="Value"
            )
            df_long.replace("OVER", np.nan, inplace=True)
            df_long.replace("", np.nan, inplace=True)
            df_long.dropna(subset=['Well', 'Value'], inplace=True, axis=0, how='all')
            df_long['Value'] = df_long['Value'].astype(float)
            df_long['Wavelength'] = df_long['Wavelength'].astype(float)
            self.data = df_long

        # single wavelength measurement
        else:
            if self.reference:
                _ = ctx.read_until(
                    lambda row: row[0]==self.label,
                    drop_empty=False
                )
                ctx.advance()
                data_block = ctx.read_until_empty_row(drop_empty=False)
                data_df = df_from_plate_like_block(data_block, data_label='value')
                
                # advance until reference
                _ = ctx.read_until(
                    lambda row: row[0]=="Reference",
                    drop_empty=False
                )
                ctx.advance()
                reference_block = ctx.read_until_empty_row(drop_empty=False)
                ref_df = df_from_plate_like_block(reference_block, data_label='reference')

                # advance until difference
                _ = ctx.read_until(
                    lambda row: row[0]=="Difference",
                    drop_empty=False
                )
                ctx.advance()
                difference_block = ctx.read_until_empty_row(drop_empty=False)
                diff_df = df_from_plate_like_block(difference_block, data_label='difference')

                self.data = (
                    pd.concat(
                        [data_df.set_index("Well"),
                        ref_df.set_index("Well"),
                        diff_df.set_index("Well")],
                        axis=1
                    )
                    .reset_index()
                )
            else:
                _ = ctx.read_until(
                    lambda row: row[0]=='<>',
                    drop_empty=False
                )
                data_block = ctx.read_until_empty_row(drop_empty=False)
                self.data = df_from_plate_like_block(data_block, data_label='value')
        
        # TODO: find end time