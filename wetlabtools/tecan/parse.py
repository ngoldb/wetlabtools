"""
This module contains code to parse data from the Tecan Excel sheets
"""

import openpyxl

def parse_header(excel_sheet):
        
        def row2list(row):
            row_list = [cell.value for cell in row]
            row_list =  [value for value in row_list if value != None and value != '' and value != ' ']
            return row_list
        
        def process_actions(iterator):
            '''Parse action list from Tecan file'''
            actions = {}
            row = next(iterator)
            row_list = row2list(row)

            while row_list != []:
                if len(row_list) == 1:
                    actions[row_list[0]] = dict()
                else:
                    actions[row_list[0]] = {'label': row_list[1]}
                row = next(iterator)
                row_list = row2list(row)
            
            for action in actions:
                row = next(iterator)
                row_list = row2list(row)
                while row_list != []:
                    actions[action][row_list[0]] = ' '.join([str(x) for x in row_list[1:]])
                    row = next(iterator)
                    row_list = row2list(row)

            return actions

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
            
            if not row_list:
                # skip empty rows
                pass

            elif 'List of actions in this measurement script:' in row_list:
                config = False
                protocol = process_actions(row_iterator)
            
            elif config:
                key, val = format_config(row_list)
                meta_data[key] = val

        return meta_data, protocol


def parse_data(excel_sheet, mode, kinetics, reference):
    """Parse data from Tecan excel file"""
    # TODO
    return None