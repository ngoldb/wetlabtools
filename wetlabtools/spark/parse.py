'''Module to parse tecan spark output'''

import openpyxl

def parse_header(excel_sheet):
        
        def row2list(row):
            row_list = [cell.value for cell in row]
            row_list =  [value for value in row_list if value != None and value != '' and value != ' ']
            return row_list
        
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