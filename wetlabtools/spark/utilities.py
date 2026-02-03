'''Utility functions, classes for Tecan Spark module'''

import re 
import numpy as np

import numpy as np
import re
import string

def row2list(row):
    row_list = [cell.value for cell in row]
    row_list =  [value for value in row_list if value != None and value != '' and value != ' ']
    return row_list

class PlateRegion:
    """
    Represents a region of a microtiter plate.
    Stores the original string, expanded wells, and a binary mask.
    Infers plate dimensions (rows × cols) from total number of wells.
    Supports ranges spanning multiple rows (e.g. A1-C3).
    """
    
    WELL_FORMATS = {
        6:  (2, 3),
        12: (3, 4),
        24: (4, 6),
        48: (6, 8),
        96: (8, 12),
        384: (16, 24),
        1536: (32, 48)
    }

    def __init__(self, region_str: str, wells_total: int = 96):
        if wells_total not in self.WELL_FORMATS:
            raise ValueError(f"Unsupported plate format: {wells_total} wells")
        
        self.region_str = region_str
        self.rows, self.cols = self.WELL_FORMATS[wells_total]
        self.row_labels = list(string.ascii_uppercase[:self.rows])
        
        self.wells = self._parse_region(region_str)
        self.mask = self._to_mask(self.wells)

    def _parse_region(self, region_str: str):
        wells = []
        parts = region_str.split(";")
        for part in parts:
            if "-" in part:
                start, end = part.split("-")
                wells.extend(self._expand_range(start.strip(), end.strip()))
            else:
                wells.append(part.strip())
        return wells

    def _expand_range(self, start: str, end: str):
        """
        Expand ranges across rows and columns, e.g. A1-C3.
        """
        row_s, col_s = re.match(r"([A-Z]+)(\d+)", start).groups()
        row_e, col_e = re.match(r"([A-Z]+)(\d+)", end).groups()

        row_s_idx = self.row_labels.index(row_s)
        row_e_idx = self.row_labels.index(row_e)

        wells = []
        for r in range(row_s_idx, row_e_idx + 1):
            for c in range(int(col_s), int(col_e) + 1):
                wells.append(f"{self.row_labels[r]}{c}")
        return wells

    def _to_mask(self, wells):
        mask = np.zeros((self.rows, self.cols), dtype=int)
        for well in wells:
            row, col = re.match(r"([A-Z]+)(\d+)", well).groups()
            r_idx = self.row_labels.index(row)
            c_idx = int(col) - 1
            if r_idx >= self.rows or c_idx >= self.cols:
                raise ValueError(f"Well {well} outside plate dimensions")
            mask[r_idx, c_idx] = 1
        return mask

    def __repr__(self):
        return f"PlateRegion('{self.region_str}', wells={len(self.wells)}, shape={self.rows}x{self.cols})"