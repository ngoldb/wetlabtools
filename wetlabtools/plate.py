import re
import string
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union



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

    def __init__(self, region_str: str, wells_total: int=96, rows: int=None, cols: int=None):
        if rows and cols:
            self.rows = rows
            self.cols = cols
            self.num_wells = rows * cols
            if self.num_wells != wells_total:
                raise ValueError(f"Total wells ({wells_total}) does not match plate format: {rows}x{cols}={self.num_wells}")
        else:
            if wells_total not in self.WELL_FORMATS:
                raise ValueError(f"Unsupported plate format: {wells_total} wells")
            self.rows, self.cols = self.WELL_FORMATS[wells_total]
            self.num_wells = wells_total
        
        self.plate_format = (self.rows, self.cols, self.num_wells)
        self.region_str = region_str
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
    

class Well:
    def __init__(self, location: str, data: Optional[Dict[str, Any]] = None):
        self.location = location  # e.g. "A1"
        self.data = data or {}

    def __repr__(self):
        return f"Well({self.location}, {self.data})"


class Plate:

    WELL_FORMATS = {
        6:  (2, 3),
        12: (3, 4),
        24: (4, 6),
        48: (6, 8),
        96: (8, 12),
        384: (16, 24),
        1536: (32, 48)
    }

    def __init__(self, num_wells: int=96, rows: int=None, cols: int=None, plate_data=None):
        if rows and cols:
            self.rows = rows
            self.cols = cols
            self.num_wells = rows * cols
        else:
            if num_wells not in self.WELL_FORMATS:
                raise ValueError(f"Unsupported plate format: {num_wells} wells")
            self.rows, self.cols = self.WELL_FORMATS[num_wells]
            self.num_wells = num_wells
        
        self.plate_format = (self.rows, self.cols, self.num_wells)
        self.row_labels = list(string.ascii_uppercase[:self.rows])
        self.plate_data = plate_data or {}
        self.wells = self._create_wells()

    def _create_wells(self):
        wells = {}
        for r, row_label in enumerate(self.row_labels):
            for c in range(1, self.cols + 1):
                loc = f"{row_label}{c}"
                wells[loc] = Well(loc)
        return wells

    def get_well(self, location: str) -> Well:
        return self.wells[location]

    # ----------------------------
    # Region-aware helpers
    # ----------------------------

    def get_region(self, region: Union["PlateRegion", str]) -> Dict[str, Well]:
        if isinstance(region, PlateRegion):
            assert self.plate_format == region.plate_format, f"incompatible format of {self} and {region}: {self.plate_format}, {region.plate_format}"

        if isinstance(region, str):
            region = PlateRegion(
                region, 
                wells_total=self.num_wells, 
                rows=self.rows, 
                cols=self.cols
            )
        return {loc: self.wells[loc] for loc in region.wells}

    def set_region_data(self, region: Union["PlateRegion", str], **kwargs):
        if isinstance(region, PlateRegion):
            assert self.plate_format == region.plate_format, f"incompatible format of {self} and {region}: {self.plate_format}, {region.plate_format}"

        if isinstance(region, str):
            region = PlateRegion(
                region, 
                wells_total=self.num_wells, 
                rows=self.rows, 
                cols=self.cols
            )

        for loc in region.wells:
            for k, v in kwargs.items():
                self.wells[loc].data[k] = v

        for k in kwargs:
            self._promote_if_identical(k)

    # ----------------------------
    # Promotion / Demotion
    # ----------------------------

    def _promote_if_identical(self, key: str):
        values = [w.data.get(key) for w in self.wells.values()]
        unique = {v for v in values if not pd.isna(v)}

        if len(unique) == 1:
            self.plate_data[key] = unique.pop()
            for w in self.wells.values():
                w.data.pop(key, None)

    def demote_plate_value(self, key: str, location: str, new_value):
        if key not in self.plate_data:
            raise KeyError(f"{key} not found at plate level")

        plate_value = self.plate_data[key]

        for w in self.wells.values():
            w.data[key] = plate_value

        self.wells[location].data[key] = new_value
        del self.plate_data[key]

    # ----------------------------
    # Import: Long format
    # ----------------------------

    @classmethod
    def from_long_dataframe(
        cls,
        df: pd.DataFrame,
        well_col="Well Location",
        plate_id_col: Optional[str] = None,
        num_wells: int = 96,
    ):
        plate = cls(num_wells=num_wells)

        for _, row in df.iterrows():
            loc = row[well_col]
            well = plate.get_well(loc)

            for col, val in row.items():
                if col in (well_col, plate_id_col):
                    continue
                if pd.notna(val):
                    well.data[col] = val

        if plate_id_col and plate_id_col in df.columns:
            unique_ids = df[plate_id_col].dropna().unique()
            if len(unique_ids) == 1:
                plate.plate_data[plate_id_col] = unique_ids[0]

        all_keys = {k for w in plate.wells.values() for k in w.data}
        for key in all_keys:
            plate._promote_if_identical(key)

        return plate

    # ----------------------------
    # Import: Plate-map format
    # ----------------------------

    @classmethod
    def from_plate_map_dataframe(
        cls,
        df: pd.DataFrame,
        value_name: str,
        num_wells: int = 96,
    ):
        plate = cls(num_wells=num_wells)

        row_labels = df.iloc[1:, 0].astype(str).tolist()
        col_labels = df.iloc[0, 1:].astype(int).tolist()

        for i, row_label in enumerate(row_labels, start=1):
            for j, col_label in enumerate(col_labels, start=1):
                val = df.iloc[i, j]
                if pd.notna(val):
                    loc = f"{row_label}{col_label}"
                    plate.wells[loc].data[value_name] = val

        plate._promote_if_identical(value_name)
        return plate

    # ----------------------------
    # Export
    # ----------------------------

    def to_long_dataframe(self, region: Union["PlateRegion", str, None] = None):
        wells = self.wells.values()

        if region is not None:
            if isinstance(region, PlateRegion):
                assert self.plate_format == region.plate_format, f"incompatible format of {self} and {region}: {self.plate_format}, {region.plate_format}"

            if isinstance(region, str):
                region = PlateRegion(
                    region,
                    wells_total=self.num_wells,
                    rows=self.rows,
                    cols=self.cols
                )
            wells = [self.wells[loc] for loc in region.wells]

        records = []
        for well in wells:
            row = {"Well Location": well.location}
            row.update(self.plate_data)
            row.update(well.data)
            records.append(row)

        return pd.DataFrame(records)

    def summary(self):
        return {
            "rows": self.rows,
            "cols": self.cols,
            "num_wells": self.num_wells,
            "plate_data": self.plate_data,
        }

    # ----------------------------
    # Export: Long format
    # ----------------------------

    def to_long_dataframe(self, region=None):
        """
        Export plate to tidy long format:
        One row per well, with plate-level and well-level metadata.
        """
        if region is not None:
            if isinstance(region, str):
                region = PlateRegion(
                    region, 
                    wells_total=self.num_wells,
                    rows=self.rows,
                    cols=self.cols
                )
            wells = [self.wells[loc] for loc in region.wells]
        else:
            wells = self.wells.values()

        records = []
        for well in wells:
            row = {"Well Location": well.location}
            row.update(self.plate_data)
            row.update(well.data)
            records.append(row)

        return pd.DataFrame(records)

    def to_long_csv(self, path, region=None, **to_csv_kwargs):
        df = self.to_long_dataframe(region=region)
        df.to_csv(path, index=False, **to_csv_kwargs)

    def to_long_excel(self, path, region=None, **to_excel_kwargs):
        df = self.to_long_dataframe(region=region)
        df.to_excel(path, index=False, **to_excel_kwargs)

    # ----------------------------
    # Export: Plate-map format
    # ----------------------------

    def to_plate_map_dataframes(self, properties=None, fill_value=np.nan):
        """
        Export one plate-map DataFrame per property.

        Returns:
            dict[property_name -> DataFrame]
        """
        if properties is None:
            properties = set(self.plate_data.keys())
            for w in self.wells.values():
                properties |= set(w.data.keys())

        maps = {}

        for prop in properties:
            df = pd.DataFrame(
                fill_value,
                index=self.row_labels,
                columns=list(range(1, self.cols + 1))
            )

            for well in self.wells.values():
                row = well.location[0]
                col = int(well.location[1:])

                if prop in well.data:
                    df.loc[row, col] = well.data[prop]
                elif prop in self.plate_data:
                    df.loc[row, col] = self.plate_data[prop]

            maps[prop] = df

        return maps

    def to_plate_map_excel(self, path, properties=None):
        """
        Write one sheet per property (great for lab use).
        """
        maps = self.to_plate_map_dataframes(properties=properties)

        with pd.ExcelWriter(path) as writer:
            for prop, df in maps.items():
                safe_name = str(prop)[:31]  # Excel sheet name limit
                df.to_excel(writer, sheet_name=safe_name)
