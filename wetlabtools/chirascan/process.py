"""
This submodule contains code to structure and process CD data
"""

import io
import numpy as np
import pandas as pd

##############################################
# restructure data into DataFrame
def data2df(data, dimensions, cells):
    """
    data: data from parse_data()
    dimension: dimesions from parse_header()

    Function to convert data fetched from chirascan csv file into a DataFrame in long format
    """
    for prop in data:
        df_tmp = pd.DataFrame()

        for prop_dim in data[prop]:
            curr_data_dim = prop_dim[0]
            lines = prop_dim[1]

            if len(curr_data_dim['ax']) == 1:
                # reading data
                df_tmp = pd.read_csv(io.StringIO(lines), header=None)
                df_tmp.columns = [curr_data_dim['ax'][0], 'value']

            else:
                # lines are the captured data as a string
                df = pd.read_csv(io.StringIO(lines))
                df.columns.values[0] = curr_data_dim['ax'][0]

                # get the number of columns expected from dimensions - required to properly melt the df
                col_dim = curr_data_dim['ax'][1]
                if col_dim in ['Cell', 'Temperature', 'Repeat']:
                    n_col_vars = dimensions[curr_data_dim['ax'][1]]

                elif col_dim == 'Wavelength':
                    wv_end = dimensions['Wavelength']['end']
                    wv_low = dimensions['Wavelength']['start']
                    step = dimensions['Wavelength']['step']
                    n_wvl = np.linspace(int(wv_low), 
                                        int(wv_end),
                                        num=(int(wv_end) - int(wv_low)) // step + 1,
                                        dtype=int)
                    n_col_vars = len(n_wvl)

                value_vars=df.columns[1:n_col_vars + 1].values

                if 'Temperature' in df.columns:
                    # check if return ramp is present
                    split_index = df.index[df['Temperature'] == max(df['Temperature'])]
                    return_ramp = all(df['Temperature'][:split_index[0] + 1].values == df['Temperature'][split_index[0]:].values[::-1])
                    
                    if return_ramp:
                        # annotating heat and cool phase and melting into long format
                        tmp_heat_df = df.iloc[:split_index[0] + 1, :].copy()
                        tmp_heat_df = tmp_heat_df.melt(
                            value_vars=value_vars,
                            id_vars=(curr_data_dim['ax'][0]),
                            var_name=curr_data_dim['ax'][1]
                        )
                        tmp_heat_df['phase'] = 'heat'
                        
                        tmp_cool_df = df.iloc[split_index[0]:, :].copy()
                        tmp_cool_df = tmp_cool_df.melt(
                            value_vars=set(str(x) for x in range(1, dimensions['Cell'] + 1)),
                            id_vars=(curr_data_dim['ax'][0]),
                            var_name=curr_data_dim['ax'][1]
                        )
                        tmp_cool_df['phase'] = 'cool'
                        df = pd.concat([tmp_heat_df, tmp_cool_df], axis='index', ignore_index=True)
                    
                    else:
                        # melting df
                        df = df.melt(
                            value_vars=value_vars,
                            id_vars=(curr_data_dim['ax'][0]),
                            var_name=curr_data_dim['ax'][1]
                        )

                    for key in curr_data_dim.keys():
                        if key != 'ax':
                            df[key.strip(':')] = curr_data_dim[key]

                elif 'Wavelength' in df.columns:
                    df = df.melt(
                        value_vars=value_vars,
                        id_vars=(curr_data_dim['ax'][0]),
                        var_name=curr_data_dim['ax'][1]
                    )

                    for key in curr_data_dim.keys():
                        if key != 'ax':
                            df[key.strip(':')] = curr_data_dim[key]
                
                df_tmp = pd.concat([df_tmp, df], axis=0, ignore_index=True)

            if len(cells) == 1:
                df_tmp['Cell'] = '1'

            data[prop] = df_tmp

    # annotating final measurement
    for prop in data.keys():
        try:
            data[prop]["is_final_measurement"] = data[prop]['Temperature'].str.contains('.1')
        except:
            data[prop]["is_final_measurement"] = False
        if any(data[prop]["is_final_measurement"]):
            data[prop]['Temperature'] = data[prop]['Temperature'].apply(lambda x: x.strip('.1'))
        
        data[prop]['Temperature'] = data[prop]['Temperature'].astype(int)

    return data