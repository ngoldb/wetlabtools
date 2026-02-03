'''Module for tecan spark experiments'''

from dataclasses import dataclass

from wetlabtools.spark.parse import parse_header, parse_action_list
from wetlabtools.spark.workflow import workflow_from_action_list

class Device:

    def __init__(self, name: str, serial_number: int, firmware: dict):
        self.name = name
        self.serial_number = serial_number
        self.firmware = firmware

    def __str__(self) -> str:
        return f"{self.name} {self.serial_number}"
    
    def __repr__(self):
        return f"Device(name={self.name}, serial_number={self.serial_number})"
    

@dataclass
class SoftwareApplication:
    __slots__ = ("name", "version")

    name: str
    version: str

    def __str__(self) -> str:
        return f"{self.name} {self.version}"


class Experiment:
    """Class for Tecan Spark experiments"""

    def __init__(self, result_file: str):
        self.file = result_file

        # parse and store meta data
        meta_data = parse_header(self.file)

        device_name = meta_data['Device'].split(' ')[0]
        serial_number = meta_data['Device'].split(' ')[-1]
        firmware = {k: v for k, v in (entry.split(":") for entry in meta_data['Firmware'].split('|'))}
        self.device = Device(device_name, serial_number, firmware)
        
        software = meta_data['Application'].split(' ')[0]
        version = meta_data['Application'].split(' ')[-1]
        self.application = SoftwareApplication(software, version)

        self.user = meta_data['User']
        self.method = meta_data["Method"]
        self.system = meta_data["System"]
        self.date = meta_data["Date"]
        self.time = meta_data["Time"]
        self.plate = meta_data["Plate"]
        self.lid_lifter = meta_data["Lid lifter"]
        self.humidity_cassette = meta_data["Humidity Cassette"]
        self.smooth_mode = meta_data["Smooth mode"]
        self.total_wells = int(self.plate.split(' ')[0][4:-3])

        # parse and populate measurement workflow
        action_list = parse_action_list(self.file)
        self.workflow = workflow_from_action_list(action_list)

    def __str__(self) -> str:
        return f"Tecan Spark Experiment from {self.date} {self.time} by {self.user}\nFile: {self.file}"
    
    def __repr__(self) -> str:
        return f'Experiment(file={self.file}, date={self.date})'