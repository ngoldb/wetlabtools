"""
This module contains code to convert plate maps from Twist into worklists for Tecan Fluent liquid handlers. Many of the settings are quite
specific to the cloning workflow and the config of the Tecan Fluent.
"""


import os 
import pandas as pd


def make_cloning_worklists(df, rxn, settings):
    """
    Function to create all required worklist files for golden gate or gibson assemblies.
    Individual worklist file for Master Mix distribution, Vector, and Insert.
    """

    # get number of reactions and vectors
    rxn_df = df.loc[df['Cloning']==rxn].copy()
    n_rxn = len(rxn_df)
    vectors = rxn_df["Vector"].unique()

    # sort dataframe by source well index and assign destination well index
    rxn_df.sort_values(by='src_numerical', ascending=True, inplace=True)
    rxn_df['dest_numerical'] = range(1, len(rxn_df) + 1)

    ## Create GWL command lines ##

    # Master Mix
    # is a regular Reagent Distribution - could be removed and replaced accordingly in FluentControl (more robust?)
    liquid_class = "MasterMix Low Volume Multi"
    aspirate_parameters = f"{settings['src_rack_label']};;;1;1"
    dispense_parameters = f"{settings['dest_rack_label']};;;1;{n_rxn}"
    distribute_command = f"R;{aspirate_parameters};{dispense_parameters};{settings['mm_volume']};{liquid_class};12;12;0\n"

    with open(settings['mm_gwl_file'], "w") as fobj:
        fobj.writelines(distribute_command)

    # Vectors
    liquid_class = "Water Contact Wet Multi low volume"
    gwl_lines = []

    for vector in vectors:
        aspirate_parameters = f"{vector};;;1;1"
        dispense_parameters = f"{settings['dest_rack_label']};;;1;{n_rxn}"
        exclude_wells = list(rxn_df.loc[rxn_df['Vector']!=vector]['dest_numerical'].values)
        if len(exclude_wells) == 0:
            distribute_command = f"R;{aspirate_parameters};{dispense_parameters};{settings['vector_volume']};{liquid_class};1;12;0\n"
        else:
            exclude_wells = ";".join([str(x) for x in exclude_wells])
            distribute_command = f"R;{aspirate_parameters};{dispense_parameters};{settings['vector_volume']};{liquid_class};1;12;0;{exclude_wells}\n"
        gwl_lines.append(distribute_command)

    with open(settings['vector_gwl_file'], "w") as fobj:
        fobj.writelines(gwl_lines)

    # Inserts
    gwl_lines = []
    liquid_class = "Water Contact Wet Single"
    for i, row in rxn_df.iterrows():
        src_well = row["src_numerical"]
        dest_well = row["dest_numerical"]
        aspirate_command = f"A;{settings['twist_label']};;;{src_well};;{settings['insert_volume']};{liquid_class};;;\n"
        dispense_command = f"D;{settings['dest_rack_label']};;;{dest_well};;{settings['insert_volume']};{liquid_class};;;\nW;\n"
        gwl_lines.append(aspirate_command)
        gwl_lines.append(dispense_command)

    with open(settings['insert_gwl_file'], "w") as fobj:
        fobj.writelines(gwl_lines)

    print(f"wrote {rxn} worklists to {os.path.dirname(settings['mm_gwl_file'])}")


def make_cloning_worklists_from_twist(plate_map_path: str, gwl_output: str, cautios: bool=True) -> pd.DataFrame:
    """
    Function to convert a plate map from Twist into worklists for cloning on the Tecan Fluent.
    A new directory will be created at the specified location containing all files required to
    run the cloning method on the Tecan Fluent. Do not rename or modify the output files.

    Returns a data frame with the expected plate maps after running the cloning on the Tecan
    
    :param plate_map_path: Path to the excel file containing the plate information from Twist
    :type plate_map_path: str
    :param gwl_output: Path to a directory where the output files will be saved
    :type gwl_output: str
    :param cautios: Whether to overwrite existing files or not
    :type cautios: bool
    """
    
    # vector names must match exactly to ensure compatibility with worktable
    ALLOWED_VECTORS = ['LM670', 'LM627', 'PHLSEC', 'PHLSEC_FC', 'CUSTOM']
    CLONING_METHODS = ['GGA', 'GIBSON']
    VERSION = 0.1
    
    # mapping alpha-numerical well indices to numerical indices
    rows = list("ABCDEFGH")
    cols = range(1, 13)
    to_numerical = {
        f"{row}{col}": (col - 1) * 8 + (rows.index(row) + 1)
        for col in cols
        for row in rows
    }

    df = pd.read_excel(plate_map_path)

    # drop entries which are not being cloned
    df.dropna(subset=['Cloning'], inplace=True)

    # map to numerical well indices
    df['src_numerical'] = df["Well Location"].map(to_numerical)
    plate_id = df["Plate ID"].unique()
    assert len(plate_id) == 1, "multiple plate barcodes"
    plate_id = plate_id[0]

    # sanity checks
    df['Cloning'] = df['Cloning'].str.upper()
    df['Vector'] = df['Vector'].str.upper()
    
    # create directory and files
    if cautios & os.path.exists(os.path.join(gwl_output, plate_id)):
        raise FileExistsError("Output directory already exists. Either disable cautios mode or delete output directory")    
    os.makedirs(os.path.join(gwl_output, plate_id), exist_ok=True)
    cloning_summary_file = os.path.join(gwl_output, plate_id, f"{plate_id}_reactions.csv")

    # summary file
    n_gga = len(df.loc[df['Cloning']=="GGA"])
    n_gib = len(df.loc[df['Cloning']=="GIBSON"])
    with open(cloning_summary_file, "w") as fobj:
        fobj.writelines(f"number_gga,number_gibson,version\n{n_gga},{n_gib},{VERSION}")
    
    # Cloning settings
    gga_settings = {
        "twist_label": "TwistPlate[001]",
        "src_rack_label": "GGA_MasterMix",
        "dest_rack_label": "96 Well PCR GGA",
        "mm_volume": 3,
        "vector_volume": 1,
        "insert_volume": 1,
        "mm_gwl_file": os.path.join(gwl_output, plate_id, f"{plate_id}_gga_mm.gwl"),
        "vector_gwl_file": os.path.join(gwl_output, plate_id, f"{plate_id}_gga_vectors.gwl"),
        "insert_gwl_file": os.path.join(gwl_output, plate_id, f"{plate_id}_gga_inserts.gwl")
    }

    gibson_settings = {
        "twist_label": "TwistPlate[001]",
        "src_rack_label": "Gibson_MasterMix",
        "dest_rack_label": "96 Well PCR Gibson",
        "mm_volume": 3,
        "vector_volume": 1,
        "insert_volume": 1,
        "mm_gwl_file": os.path.join(gwl_output, plate_id, f"{plate_id}_gib_mm.gwl"),
        "vector_gwl_file": os.path.join(gwl_output, plate_id, f"{plate_id}_gib_vectors.gwl"),
        "insert_gwl_file": os.path.join(gwl_output, plate_id, f"{plate_id}_gib_inserts.gwl")
    }

    if n_gga > 0:
        make_cloning_worklists(df, "GGA", gga_settings)

    if n_gib > 0:
        make_cloning_worklists(df, "GIBSON", gibson_settings)

    return df