import datetime
from wetlabtools.spark.actions.base_action import Action
from wetlabtools.spark.action_registry import register_action

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
