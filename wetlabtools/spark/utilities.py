'''Utility functions, classes for Tecan Spark module'''

import numpy as np

def row2list(row):
    row_list = [cell.value for cell in row]
    row_list =  [value for value in row_list if value != None and value != '' and value != ' ']
    return row_list
