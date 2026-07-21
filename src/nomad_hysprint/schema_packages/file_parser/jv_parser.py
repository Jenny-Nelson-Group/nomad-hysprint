#
# Copyright The NOMAD Authors.
#
# This file is part of NOMAD. See https://nomad-lab.eu for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import ast
import json
import re
from datetime import datetime
from io import BytesIO, StringIO

import numpy as np
import pandas as pd
from baseclasses.helper.utilities import convert_datetime


def get_jv_data_hysprint(filedata, filename=None):
    # Block to clean up some bad characters found in the file which gives
    # trouble reading.

    filedata = filedata.replace('²', '^2')

    df = pd.read_csv(
        StringIO(filedata),
        skiprows=8,
        nrows=9,
        sep='\t',
        index_col=0,
        engine='python',
        encoding='unicode_escape',
    )
    df_header = pd.read_csv(
        StringIO(filedata),
        skiprows=1,
        nrows=6,
        header=None,
        sep=':|\t',
        index_col=0,
        encoding='unicode_escape',
        engine='python',
    )
    df_curves = pd.read_csv(
        StringIO(filedata),
        header=19,
        skiprows=[20],
        sep='\t',
        encoding='unicode_escape',
        engine='python',
    )
    df_curves = df_curves.dropna(how='all', axis=1)

    df_header.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
    df.replace([np.inf, -np.inf, np.nan], 0, inplace=True)

    number_of_curves = len(df_curves.columns) - 1

    jv_dict = {}
    jv_dict['active_area'] = df_header.iloc[0, 1]
    jv_dict['intensity'] = df_header.iloc[1, 1]
    jv_dict['integration_time'] = df_header.iloc[2, 1]
    jv_dict['settling_time'] = df_header.iloc[3, 1]
    jv_dict['averaging'] = df_header.iloc[4, 1]
    jv_dict['compliance'] = df_header.iloc[5, 1]

    jv_dict['J_sc'] = list(abs(df.iloc[0]))[:number_of_curves]
    jv_dict['V_oc'] = list(df.iloc[1])[:number_of_curves]
    jv_dict['Fill_factor'] = list(df.iloc[2])[:number_of_curves]
    jv_dict['Efficiency'] = list(df.iloc[3])[:number_of_curves]
    jv_dict['P_MPP'] = list(df.iloc[4])[:number_of_curves]
    jv_dict['J_MPP'] = list(abs(df.iloc[5]))[:number_of_curves]
    jv_dict['U_MPP'] = list(df.iloc[6])[:number_of_curves]
    jv_dict['R_ser'] = list(df.iloc[7])[:number_of_curves]
    jv_dict['R_par'] = list(df.iloc[8])[:number_of_curves]

    jv_dict['jv_curve'] = []
    for column in range(1, len(df_curves.columns)):
        jv_dict['jv_curve'].append(
            {
                'name': df_curves.columns[column],
                'voltage': df_curves[df_curves.columns[0]].values,
                'current_density': df_curves[df_curves.columns[column]].values,
            }
        )

    # Add Last Modified Date from filename if possible
    match = re.search(r'\.(\d{10})_', filename)
    if match:
        epoch = int(match.group(1))
        dt = datetime.fromtimestamp(epoch)
        formatted = dt.strftime('%d.%m.%Y %H:%M:%S')
        print(formatted)
        jv_dict['datetime'] = formatted

    return jv_dict


def get_jv_data_pvcomb_1(filedata):
    # Block to clean up some bad characters found in the file which gives
    # trouble reading.

    filedata = filedata.replace('²', '^2')

    df = pd.read_csv(
        StringIO(filedata),
        header=None,
        skiprows=3,
        nrows=6,
        sep=':\t',
        index_col=0,
        engine='python',
        encoding='unicode_escape',
    )
    df_header = pd.read_csv(
        StringIO(filedata),
        skiprows=14,
        nrows=12,
        header=None,
        sep=':\t',
        index_col=0,
        encoding='unicode_escape',
        engine='python',
    )
    df_curves = pd.read_csv(
        StringIO(filedata),
        header=26,
        # skiprows=[20],
        sep='\t',
        encoding='unicode_escape',
        engine='python',
    )

    df_curves = df_curves.dropna(how='all', axis=1)

    df_header.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
    df.replace([np.inf, -np.inf, np.nan], 0, inplace=True)

    jv_dict = {}
    jv_dict['datetime'] = convert_datetime(df.iloc[0, 0], '%d.%m.%Y %H:%M:%S')
    jv_dict['active_area'] = float(df.iloc[1, 0])
    jv_dict['intensity'] = float(df.iloc[2, 0]) / 10
    jv_dict['compliance'] = float(df.iloc[4, 0]) * 1000 / float(df.iloc[1, 0])
    jv_dict['settling_time'] = float(df.iloc[3, 0])

    jv_dict['J_sc'] = [df_header.iloc[7, 0]]
    jv_dict['V_oc'] = [df_header.iloc[1, 0]]
    jv_dict['Fill_factor'] = [float(df_header.iloc[5, 0])]
    jv_dict['Efficiency'] = [df_header.iloc[6, 0]]
    jv_dict['J_MPP'] = [df_header.iloc[8, 0]]
    jv_dict['U_MPP'] = [df_header.iloc[3, 0]]
    jv_dict['R_ser'] = [df_header.iloc[10, 0]]
    jv_dict['R_par'] = [df_header.iloc[11, 0]]

    jv_dict['jv_curve'] = [
        {
            'name': df.iloc[5, 0],
            'voltage': df_curves['V [V]'].values,
            'current_density': df_curves['J [mA/cm2]'].values,
        }
    ]

    return jv_dict


def get_jv_data_iris(filedata):
    # Block to clean up some bad characters found in the file which gives
    # trouble reading.

    filedata = filedata.replace('²', '^2')

    df = pd.read_csv(
        StringIO(filedata),
        skiprows=30,
        header=0,
        nrows=11,
        sep='\t',
        index_col=0,
        engine='python',
        encoding='unicode_escape',
    )

    df_header = pd.read_csv(
        StringIO(filedata),
        skiprows=0,
        nrows=29,
        header=None,
        sep='\t',
        index_col=0,
        encoding='unicode_escape',
        engine='python',
    )  # , on_bad_lines=lambda x: x[:2])

    df_curves = pd.read_csv(
        StringIO(filedata),
        header=0,
        skiprows=43,
        sep='\t',
        encoding='unicode_escape',
        engine='python',
    )
    df_curves = df_curves.dropna(how='all', axis=1)

    df_header.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
    df.replace([np.inf, -np.inf, np.nan], 0, inplace=True)

    jv_dict = {}
    jv_dict['datetime'] = convert_datetime(df_header.iloc[0, 0], '%H:%M:%S - %d.%m.%Y')
    jv_dict['active_area'] = list(ast.literal_eval(df_header.iloc[12, 0]))[0]
    jv_dict['intensity'] = float(df_header.iloc[15, 0])  # Power Density
    jv_dict['integration_time'] = float(df_header.iloc[9, 0]) * 1000
    jv_dict['settling_time'] = float(df_header.iloc[10, 0]) * 1000
    jv_dict['averaging'] = float(df_header.iloc[8, 0])
    # jv_dict['compliance'] = df_header.iloc[5, 1]
    jv_dict['J_sc'] = list(abs(df.iloc[1].astype(np.float64)))
    jv_dict['V_oc'] = list(df.iloc[0].astype(np.float64))
    jv_dict['Fill_factor'] = list(df.iloc[2].astype(np.float64))
    jv_dict['Efficiency'] = list(df.iloc[3].astype(np.float64))
    jv_dict['P_MPP'] = list(abs(df.iloc[6].astype(np.float64)))
    jv_dict['J_MPP'] = list(abs(df.iloc[5].astype(np.float64)))
    jv_dict['U_MPP'] = list(df.iloc[4].astype(np.float64))
    jv_dict['R_ser'] = list(df.iloc[7].astype(np.float64))
    jv_dict['R_par'] = list(df.iloc[8].astype(np.float64))

    jv_dict['jv_curve'] = []

    for column in range(0, len(df_curves.columns) // 2):
        jv_dict['jv_curve'].append(
            {
                'name': '_'.join(df_curves.columns[2 * column].split('_')[-3:]),
                'dark': True if 'dark' in df_curves.columns[2 * column].lower() else False,
                'voltage': df_curves[df_curves.columns[2 * column]].values,
                'current_density': df_curves[df_curves.columns[2 * column + 1]].values,
            }
        )

    return jv_dict


def get_jv_data_iris_json(filedata):
    data = json.loads(filedata)
    jv_dict = {}
    jv_dict['intensity'] = data['parameters']['irradiance'] / 10
    jv_dict['settling_time'] = data['parameters']['settling_time_ms']
    jv_dict['jv_curve'] = []
    for c in ['J_sc', 'V_oc', 'Fill_factor', 'Efficiency', 'P_MPP', 'J_MPP', 'U_MPP', 'R_ser', 'R_par']:
        jv_dict[c] = []
    for m in data['data']:
        jv_dict['J_sc'].append(m['measurements'][0]['Jsc [mA/cm2]'])
        jv_dict['V_oc'].append(m['measurements'][0]['Voc [V]'])
        jv_dict['Fill_factor'].append(m['measurements'][0]['FF [%]'])
        jv_dict['Efficiency'].append(m['measurements'][0]['Eff_1000 [%]'])
        jv_dict['J_MPP'].append(m['measurements'][0]['J_mpp [mA/cm2]'])
        jv_dict['U_MPP'].append(m['measurements'][0]['V_mpp [V]'])
        jv_dict['R_ser'].append(m['measurements'][0]['Rs [\u2126cm2]'])
        jv_dict['R_par'].append(m['measurements'][0]['Rp [\u2126cm2]'])
        jv_dict['jv_curve'].append(
            {
                'name': m['cell'] + ' ' + m['direction'],
                'voltage': [v['V_meas [V]'] for v in m['measurements']],
                'current_density': [c['J [mA/cm2]'] for c in m['measurements']],
            }
        )

    return jv_dict


def get_jv_solar_sim_jn(filedata, nmbr_of_pixels=6):
    filedata = filedata.replace('²', '^2')

    df_header = pd.read_csv(
        StringIO(filedata),
        skiprows=0,
        nrows=29,
        header=None,
        sep='\t',
        index_col=0,
        encoding='unicode_escape',
        engine='python',
    )

    df_fom = pd.read_csv(
        StringIO(filedata),
        skiprows=30,
        header=0,
        nrows=11,
        sep='\t',
        index_col=0,
        engine='python',
        encoding='unicode_escape',
    )

    df_curves = pd.read_csv(
        StringIO(filedata),
        skiprows=43,
        header=0,
        sep='\t',
        encoding='unicode_escape',
        engine='python',
    )

    df_curves = df_curves.dropna(how='all', axis=1)

    df_header.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
    df_fom.replace([np.inf, -np.inf, np.nan], 0, inplace=True)

    jv_dict = {}
    jv_dict['datetime'] = convert_datetime(df_header.iloc[0, 0], '%H:%M:%S - %d.%m.%Y')
    jv_dict['active_area'] = list(ast.literal_eval(df_header.iloc[12, 0]))[0]
    jv_dict['intensity'] = float(df_header.iloc[15, 0])
    jv_dict['V_oc'] = list(df_fom.iloc[0].astype(np.float64))
    jv_dict['J_sc'] = list(abs(df_fom.iloc[1].astype(np.float64)))
    jv_dict['Fill_factor'] = list(df_fom.iloc[2].astype(np.float64))
    jv_dict['Efficiency'] = list(df_fom.iloc[3].astype(np.float64))
    jv_dict['U_MPP'] = list(df_fom.iloc[4].astype(np.float64))
    jv_dict['J_MPP'] = list(abs(df_fom.iloc[5].astype(np.float64)))
    jv_dict['R_ser'] = list(df_fom.iloc[7].astype(np.float64))
    jv_dict['R_par'] = list(df_fom.iloc[8].astype(np.float64))

    jv_dict['jv_curve'] = []

    for column in range(0, len(df_curves.columns) // 2):
        jv_dict['jv_curve'].append(
            {
                'name': '_'.join(df_curves.columns[2 * column].split('_')[-3:]),
                'voltage': df_curves[df_curves.columns[2 * column]].values,
                'current_density': df_curves[df_curves.columns[2 * column + 1]].values,
            }
        )

    return jv_dict


def find_max_power_point(voltage, current_density):
    """Return (voltage, current_density) magnitudes at the max power point of a jv curve."""
    voltage = np.asarray(voltage, dtype=np.float64)
    current_density = np.asarray(current_density, dtype=np.float64)
    power = voltage * current_density
    generating = power < 0
    if generating.any():
        candidates = np.where(generating)[0]
        idx = candidates[np.argmax(np.abs(power[candidates]))]
    else:
        idx = int(np.argmin(power))
    return abs(voltage[idx]), abs(current_density[idx])


def calculate_resistances(voltage, current_density, voc, window=3):
    """
    Series resistance from the local dV/dJ slope of the curve at V=voc,
    shunt resistance from the local dV/dJ slope at V=0. current_density
    must be in mA/cm^2; result is in ohm*cm^2.
    """
    voltage = np.asarray(voltage, dtype=np.float64)
    current_density = np.asarray(current_density, dtype=np.float64)
    order = np.argsort(voltage)
    voltage = voltage[order]
    current_density = current_density[order]

    def local_slope(idx):
        lo = max(idx - window, 0)
        hi = min(idx + window + 1, len(voltage))
        slope, _ = np.polyfit(current_density[lo:hi], voltage[lo:hi], 1)
        return abs(slope) * 1000  # V/(mA/cm^2) -> ohm*cm^2

    voc_idx = int(np.argmin(np.abs(voltage - voc)))
    jsc_idx = int(np.argmin(np.abs(voltage)))

    return local_slope(voc_idx), local_slope(jsc_idx)


def get_jv_data_solar_sim_jn_excel(filedata):
    """Parses the JN solar simulator '<sample> Light Data.xlsx' export."""
    df = pd.read_excel(BytesIO(filedata), sheet_name=0, header=None)

    jv_dict = {}
    jv_dict['datetime'] = df.iloc[0, 1]
    jv_dict['active_area'] = float(df.iloc[1, 1]) / 100  # mm^2 -> cm^2

    n_pixels = 0
    row = 5
    while row < len(df) and isinstance(df.iloc[row, 0], str) and df.iloc[row, 0].startswith('Pixel'):
        n_pixels += 1
        row += 1

    v_oc = [float(df.iloc[5 + p, 2]) for p in range(n_pixels)]
    jv_dict['V_oc'] = v_oc
    jv_dict['J_sc'] = [abs(float(df.iloc[5 + p, 1])) * 1000 for p in range(n_pixels)]  # A/cm^2 -> mA/cm^2
    jv_dict['Fill_factor'] = [float(df.iloc[5 + p, 3]) for p in range(n_pixels)]
    jv_dict['Efficiency'] = [float(df.iloc[5 + p, 4]) for p in range(n_pixels)]

    curve_header_row = 5 + n_pixels + 1
    df_curves = pd.read_excel(BytesIO(filedata), sheet_name=0, header=curve_header_row)
    df_curves = df_curves.dropna(how='all', axis=1)

    jv_dict['jv_curve'] = []
    u_mpp, j_mpp, r_ser, r_par = [], [], [], []
    for pixel in range(n_pixels):
        voltage = df_curves.iloc[:, 2 * pixel].astype(np.float64).values
        current_density = df_curves.iloc[:, 2 * pixel + 1].astype(np.float64).values * 1000  # A/cm^2 -> mA/cm^2

        jv_dict['jv_curve'].append(
            {
                'name': f'Pixel {pixel + 1}',
                'voltage': voltage,
                'current_density': current_density,
            }
        )

        mpp_v, mpp_j = find_max_power_point(voltage, current_density)
        u_mpp.append(mpp_v)
        j_mpp.append(mpp_j)

        ser, par = calculate_resistances(voltage, current_density, v_oc[pixel])
        r_ser.append(ser)
        r_par.append(par)

    jv_dict['U_MPP'] = u_mpp
    jv_dict['J_MPP'] = j_mpp
    jv_dict['R_ser'] = r_ser
    jv_dict['R_par'] = r_par

    return jv_dict


def get_jv_data(filedata, filename=None):
    if filename and filename.lower().endswith('.xlsx'):
        return get_jv_data_solar_sim_jn_excel(filedata), 'JN Solar Simulator'
    if filedata.startswith('Keithley'):
        return get_jv_data_hysprint(filedata, filename), 'HySprint HyVap'
    if 'SoSim PVcomB' in filedata:
        return get_jv_data_pvcomb_1(filedata), 'PVcomB'
    if filedata.startswith('{') and 'parameters' in filedata[:50]:
        return get_jv_data_iris_json(filedata), 'IRIS'
    if 'Location' in filedata[:200] and 'Sample' in filedata[:200] and 'User' in filedata[:200]:
        return get_jv_data_iris(filedata), 'IRIS HZBGloveBoxes Pero4SOSIMStorage'
    return None, None
