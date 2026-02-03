'''Module to parse tecan spark output'''

import openpyxl
import pandas as pd
from wetlabtools.spark.utilities import row2list


def parse_header(excel_sheet):
        
    def format_config(row_list):
        '''formatting system config data from tecan file'''

        if len(row_list) == 1 and 'Method name:' in row_list[0]:
            key = 'Method'
            val = row_list[0].strip('Method name: ')
        elif 'Device:' in row_list[0] or 'Application:' in row_list[0]:
            key = row_list[0].split(': ')[0]
            val = ' '.join([row_list[0].split(': ')[1], row_list[1]])
        elif len(row_list) == 2:
            key = row_list[0]
            val = row_list[1]
        else:
            key = row_list[0]
            val = row_list[1:]

        return (key.strip(':'), val)
        
    wb_obj = openpyxl.load_workbook(excel_sheet)
    sheet_obj = wb_obj.active

    meta_data = dict()
    row_iterator = iter(sheet_obj.rows)
    config = True

    for row in row_iterator:
        row_list = row2list(row)
        
        # skip empty rows
        if not row_list:
            pass
        
        # end if hitting protocol section
        elif 'List of actions in this measurement script:' in row_list:
            break
        
        elif config:
            key, val = format_config(row_list)
            meta_data[key] = val

    return meta_data


def parse_action_list(filepath: str, sheet_name=0):
    """
    Reads the Excel file and extracts the block under
    'List of actions in this measurement script:'.
    Uses the column index of the first non-empty cell as indentation.
    Stops at the first empty row after the block.

    Returns a list of (indent_level, [action, label]) tuples.
    """
    df = pd.read_excel(filepath, sheet_name=sheet_name, header=None)
    
    actions = []
    start = False
    
    for row in df.itertuples(index=False):
        cells = [c if pd.notna(c) else "" for c in row]
        # join all non-empty cells for content, but track first non-empty cell index
        if not start:
            if any("List of actions in this measurement script" in str(c) for c in cells):
                start = True
            continue
        
        if start:
            if all(c == "" for c in cells):  # stop at first empty row
                break
            
            # first non-empty cell determines indent level
            for idx, cell in enumerate(cells):
                if cell != "":
                    indent_level = idx
                    break
            
            # join remaining cells into one string
            content = [str(c).lower() for c in cells if c != ""]
            actions.append((indent_level, content))
    
    return actions
