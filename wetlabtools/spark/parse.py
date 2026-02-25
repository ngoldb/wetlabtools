'''Module to parse tecan spark output'''

import openpyxl
import numpy as np
import pandas as pd
from typing import List, Any, Optional
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

        return (key.strip().strip(':'), val)
        
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
            meta_data[key.strip()] = val

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
            content = [str(c) for c in cells if c != ""]
            actions.append((indent_level, content))
    
    return actions


class ParseContext:
    """
    Sequential parsing of excel file

    Parameters
    ----------------
    df: pd.DataFrame
        input dataframe from excel file. No headers

    drop_empty_cells: bool
        If True only cells with values will be returned. Default: False
    """

    def __init__(self, df: pd.DataFrame, drop_empty_cells: bool=False):
        self.rows: List[List[Any]] = (
            df.fillna("").astype(str).values.tolist()
        )
        self.cursor: int = 0
        self.total_rows: int = len(self.rows)
        self.drop_empty_cells = drop_empty_cells

    # -------------------------
    # Internal row formatting
    # -------------------------

    def _format_row(
        self,
        row: List[str],
        drop_empty: Optional[bool]=None
    ) -> List[str]:

        if drop_empty is None:
            drop_empty = self.drop_empty_cells

        if drop_empty:
            row = [cell for cell in row if cell != ""]

        return row
    
    # -------------------------
    # Core navigation
    # -------------------------

    def eof(self) -> bool:
        return self.cursor >= self.total_rows

    def current(self, drop_empty: Optional[bool]=None) -> List[str]:
        if self.eof():
            raise IndexError("ParseContext: cursor out of bounds")
        row = self.rows[self.cursor]
        return self._format_row(row, drop_empty)

    def peek(self, offset: int=0, drop_empty: Optional[bool]=None) -> Optional[List[str]]:
        idx = self.cursor + offset
        if 0 <= idx < self.total_rows:
            return self._format_row(self.rows[idx], drop_empty)
        return None

    def advance(self, n: int=1) -> None:
        self.cursor += n
        if self.cursor > self.total_rows:
            raise IndexError("ParseContext: advanced beyond file")

    # -------------------------
    # Convenience helpers
    # -------------------------

    def skip_empty_rows(self) -> None:
        while not self.eof() and all(cell == "" for cell in self.current()):
            self.advance()

    def read_until_empty_row(self, drop_empty: Optional[bool]=None) -> List[List[str]]:
        """
        Read consecutive non-empty rows.
        Stops at first fully empty row.
        """
        block = []
        while not self.eof() and not self.current(drop_empty=True)==[]:
            block.append(self.current(drop_empty))
            self.advance()
        return block

    def read_while(self, condition, drop_empty: Optional[bool]=None) -> List[List[str]]:
        """
        Read rows while condition(row) is True.
        """
        block = []
        while not self.eof() and condition(self.current(drop_empty)):
            block.append(self.current(drop_empty))
            self.advance()
        return block

    def read_until(self, condition, drop_empty: Optional[bool]=None) -> List[List[str]]:
        """
        Read rows until condition(row) is True.
        """
        block = []
        while not self.eof() and not condition(self.current(drop_empty)):
            block.append(self.current(drop_empty))
            self.advance()
        return block
    

def block_2_dict(block):
    """creates a dict from a block (list of rows)"""
    return {
        row[0]: row[1] if len(row)==2 else row[1:] 
        for row in block
    }


# ================================================
# Functions to create data frames from data blocks
# ================================================

def df_from_plate_like_block(data_block, data_label: str='Value'):
    """Melt single value data from a plate-like format into a long format df"""
    
    df = pd.DataFrame(
        [r[1:] for r in data_block[1:]],
        columns=data_block[0][1:],
        index=[r[0] for r in data_block[1:]]
    )

    try:
        df.drop(labels='', axis=1, inplace=True)
    except KeyError: pass
    
    df.columns = df.columns.astype(float).astype(int)
    df_long = df.stack().reset_index()
    df_long.columns = ["Row", "Column", data_label]
    df_long["Well"] = df_long["Row"] + df_long["Column"].astype(str)
    df_long = df_long[["Well", data_label]]
    df_long[data_label].replace('OVER', np.nan, inplace=True)
    df_long[data_label].replace('', np.nan, inplace=True)
    df_long[data_label] = df_long[data_label].astype(float)

    return df_long


def df_from_multiple_reads_kinetic(data_block):
    well = data_block[0][0]
    df = pd.DataFrame(
        [r[1:] for r in data_block[1:]],
        columns=data_block[0][1:],
        index=[r[0] for r in data_block[1:]]
    )
    df = df.T
    df.index.name = "Cycle"
    df = df.reset_index()

    df["Well"] = well

    # data types
    df["Mean"] = df["Mean"].astype(float)

    for col_name in df.columns:
        if col_name == "Well":
            pass
        else:
            df[col_name] = df[col_name].astype(float)
    
    df['Cycle'] = df['Cycle'].astype(int)
    
    return df


# ========================
# Error handling
# ========================

class ParseError(Exception):
    """Base class for parsing-related errors."""
    pass


class BlockMismatchError(ParseError):

    def __init__(self, action, cursor):
        super().__init__(
            f"{action} failed to parse at row {cursor}. Block format did not match expected format"
        )
    