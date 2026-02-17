"""
This module contains code to convert plate maps from Twist into worklists for Tecan Fluent liquid handlers. Many of the settings are quite
specific to the cloning workflow and the config of the Tecan Fluent.
"""


import os 
import pandas as pd

from wetlabtools.plate import Plate

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
    rxn_df['dest_label'] = settings['dest_rack_label']

    ## Create GWL command lines ##

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
    return rxn_df


def make_transformation_worklist(dfs: list[pd.DataFrame], settings):

    cell_gwl_lines = []
    dna_gwl_lines = []

    for transform_df in dfs:
        # distribute strains
        dest_rack_label = transform_df['dest_label'].unique()[0]
        n_transforms = len(transform_df)

        for strain in transform_df['Transform'].unique():
            exclude_wells = list(transform_df.loc[transform_df['Transform']!=strain]['dest_numerical'].values)
            
            aspirate_parameters = f"{strain};;;1;1"
            dispense_parameters = f"{dest_rack_label};;;1;{n_transforms}"
            distribute_command = f"R;{aspirate_parameters};{dispense_parameters};{settings['cell_volume']};;1;12;0"

            if len(exclude_wells) == 0:
                distribute_command = distribute_command + '\n'
            else:
                exclude_wells = ";".join([str(x) for x in exclude_wells])
                distribute_command = distribute_command + f";{exclude_wells}\n"
            cell_gwl_lines.append(distribute_command)

        # add dna to cells
        for i, row in transform_df.iterrows():
            src_well = row["src_numerical"]
            dest_well = row["dest_numerical"]
            src_rack_label = row["src_label"]
            dest_rack_label = row["dest_label"]

            aspirate_command = f"A;{src_rack_label};;;{src_well};;{settings['dna_volume']};;;;\n"
            dispense_command = f"D;{dest_rack_label};;;{dest_well};;{settings['dna_volume']};;;;\nW;\n"
            dna_gwl_lines.append(aspirate_command)
            dna_gwl_lines.append(dispense_command)

    with open(settings["cell_distribution_gwl"], "w") as fobj:
        fobj.writelines(cell_gwl_lines)

    with open(settings["dna_distribution_gwl"], "w") as fobj:
        fobj.writelines(dna_gwl_lines)

    print(f"wrote transformation worklists to {os.path.dirname(settings['cell_distribution_gwl'])}")
    return


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
    ALLOWED_VECTORS = ['LM670', 'LM627', 'PHLSEC', 'PHLSEC_FC', 'CUSTOM_1', 'CUSTOM_2', 'CUSTOM_3']
    CLONING_METHODS = ['GGA', 'GIBSON']
    TRANSFORMATION_STRAINS = ['HB101', 'T7EXPRESS', 'NEBSTABLE', 'DH5A', 'BL21', 'CUSTOM_1', 'CUSTOM_2', 'CUSTOM_3']
    VERSION = 0.1
    
    # mapping alpha-numerical well indices to numerical indices
    rows = list("ABCDEFGH")
    cols = range(1, 13)
    to_numerical = {
        f"{row}{col}": (col - 1) * 8 + (rows.index(row) + 1)
        for col in cols
        for row in rows
    }
    to_alphanumerical = {v: k for k, v in to_numerical.items()}

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
    assert df['Cloning'].isin(CLONING_METHODS).all(), f"Unknown cloning method(s): {list(df.loc[~df['Cloning'].isin(CLONING_METHODS)]['Cloning'].unique())}. Cloning methods must be {CLONING_METHODS}"
    assert df['Vector'].isin(ALLOWED_VECTORS).all(), f"Unknown vector: {list(df.loc[~df['Vector'].isin(ALLOWED_VECTORS)]['Vector'].unique())}. Vectors must be {ALLOWED_VECTORS}"
    
    # create directory and files
    if cautios & os.path.exists(os.path.join(gwl_output, plate_id)):
        raise FileExistsError("Output directory already exists. Either disable cautios mode or delete output directory")    
    os.makedirs(os.path.join(gwl_output, plate_id), exist_ok=True)
    cloning_summary_file = os.path.join(gwl_output, plate_id, f"{plate_id}_reactions.csv")
    
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

    n_gga = len(df.loc[df['Cloning']=="GGA"])
    n_gib = len(df.loc[df['Cloning']=="GIBSON"])

    if n_gga > 0:
        gga_df = make_cloning_worklists(df, "GGA", gga_settings)
    else:
        gga_df = pd.DataFrame()

    if n_gib > 0:
        gib_df = make_cloning_worklists(df, "GIBSON", gibson_settings)
    else:
        gib_df = pd.DataFrame()

    # ensure order from Twist plate
    cloning_df = pd.concat([gga_df, gib_df], ignore_index=True)
    cloning_df.sort_values(by='src_numerical', inplace=True)
    
    ### TRANSFORMATION ###
    transformations = []
    transformation_df = cloning_df.copy()
    transformation_df.dropna(subset=["Transform_1", "Transform_2"], how='all', inplace=True) 
    transformation_df.drop('src_numerical', axis=1, inplace=True)
    transformation_df.rename({'dest_numerical': 'src_numerical', 'dest_label': 'src_label'}, inplace=True, axis=1)

    # first plate is cloning strain
    transform_1_df = transformation_df.copy()
    transform_1_df.dropna(subset="Transform_1", inplace=True)
    n_transform_1 = len(transform_1_df)
    if n_transform_1 > 0:
        transform_1_df['Transform_1'] = transform_1_df['Transform_1'].str.upper()
        assert transform_1_df['Transform_1'].isin(TRANSFORMATION_STRAINS).all(), f"Unknown transformation strain: {list(transformation_df.loc[~transformation_df['Transform_1'].isin(TRANSFORMATION_STRAINS)]['Transform_1'].unique())}. Transformation strains must be {TRANSFORMATION_STRAINS}"
        transform_1_df['dest_label'] = "96 Well PCR Transform 1"
        transform_1_df['dest_numerical'] = range(1, len(transform_1_df)+1)
        transform_1_df.rename({'Transform_1': 'Transform',}, inplace=True, axis=1)
        transform_1_df.drop('Transform_2', inplace=True, axis=1)
        transformations.append(transform_1_df)
        
    # second plate is expression strain
    transform_2_df = transformation_df.copy()
    transform_2_df.dropna(subset="Transform_2", inplace=True)
    n_transform_2 = len(transform_2_df)
    if n_transform_2 > 0:
        transform_2_df['Transform_2'] = transform_2_df['Transform_2'].str.upper()
        assert transform_2_df['Transform_2'].isin(TRANSFORMATION_STRAINS).all(), f"Unknown transformation strain: {list(transformation_df.loc[~transformation_df['Transform_2'].isin(TRANSFORMATION_STRAINS)]['Transform_2'].unique())}. Transformation strains must be {TRANSFORMATION_STRAINS}"
        transform_2_df['dest_label'] = "96 Well PCR Transform 2"
        transform_2_df['dest_numerical'] = range(1, len(transform_2_df)+1)
        transform_2_df.rename({'Transform_2': 'Transform',}, inplace=True, axis=1)
        transform_2_df.drop('Transform_1', inplace=True, axis=1)
        transformations.append(transform_2_df)
        
    # create gwl files
    transform_settings = {
        "cell_volume": 20,
        "dna_volume": 2,
        "cell_distribution_gwl": os.path.join(gwl_output, plate_id, f"{plate_id}_distribute_cells.gwl"),
        "dna_distribution_gwl": os.path.join(gwl_output, plate_id, f"{plate_id}_distribute_dna.gwl")
    }
    if len(transformations) > 0:
        make_transformation_worklist(transformations, transform_settings)

    # summary file
    with open(cloning_summary_file, "w") as fobj:
        fobj.writelines(f"number_gga,number_gibson,n_transform_1,n_transform_2,version\n{n_gga},{n_gib},{n_transform_1},{n_transform_2},{VERSION}")

    # Plate maps for users
    if n_gga > 0:
        gga_map_xlsx = os.path.join(gwl_output, plate_id, f"{plate_id}_gga_plate_map.xlsx")
        gga_map = gga_df[['Name', 'Vector', 'dest_numerical']].copy()
        gga_map['Well Location'] = gga_map['dest_numerical'].map(to_alphanumerical)
        gga_map.drop('dest_numerical', inplace=True, axis=1)
        plate = Plate.from_long_dataframe(gga_map)
        plate.to_plate_map_excel(gga_map_xlsx)

    if n_gib > 0:
        gib_map_xlsx = os.path.join(gwl_output, plate_id, f"{plate_id}_gibson_plate_map.xlsx")
        gib_map = gib_df[['Name', 'Vector', 'dest_numerical']].copy()
        gib_map['Well Location'] = gib_map['dest_numerical'].map(to_alphanumerical)
        gib_map.drop('dest_numerical', inplace=True, axis=1)
        plate = Plate.from_long_dataframe(gib_map)
        plate.to_plate_map_excel(gib_map_xlsx)

    if n_transform_1 > 0:
        trf_1_map_xlsx = os.path.join(gwl_output, plate_id, f"{plate_id}_transform_1_plate_map.xlsx")
        trf_1_map = transform_1_df[['Name', 'Transform', 'dest_numerical']].copy()
        trf_1_map['Well Location'] = trf_1_map['dest_numerical'].map(to_alphanumerical)
        trf_1_map.rename({'Transform': 'Strain'}, inplace=True, axis=1)
        trf_1_map.drop('dest_numerical', inplace=True, axis=1)
        plate = Plate.from_long_dataframe(trf_1_map)
        plate.to_plate_map_excel(trf_1_map_xlsx)
    
    if n_transform_2 > 0:
        trf_2_map_xlsx = os.path.join(gwl_output, plate_id, f"{plate_id}_transform_2_plate_map.xlsx")
        trf_2_map = transform_1_df[['Name', 'Transform', 'dest_numerical']].copy()
        trf_2_map['Well Location'] = trf_2_map['dest_numerical'].map(to_alphanumerical)
        trf_2_map.rename({'Transform': 'Strain'}, inplace=True, axis=1)
        trf_2_map.drop('dest_numerical', inplace=True, axis=1)
        plate = Plate.from_long_dataframe(trf_2_map)
        plate.to_plate_map_excel(trf_2_map_xlsx)

    return df, cloning_df, transform_1_df, transform_2_df