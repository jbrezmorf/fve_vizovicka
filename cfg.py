# Liberec
location = dict(
    latitude=50.7441414,
    longitude=15.0217061
)

# Fixed parameters of pvsystem.FixedMount, https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.pvsystem.FixedMount.html#pvlib.pvsystem.FixedMount
mount = dict(
    racking_model='open_rack',  #   0: open-rack, 1: roof-mount, 2: gcr
    module_height=0.5,          # [m], height above ground
)

inverter = {
    'Name': 'GoodWe GW15K-ET',
    'Paco': 15000,  # Maximum AC output power in watts
    'Pdco': 15300,  # DC power input at which Paco is achieved in watts
    'Vdco': 620,  # DC voltage at which Paco is achieved in volts
    'Pso': 30,  # DC power required to start the inverter in watts
    'C0': -1.1e-5,  # Coefficient for the quadratic term of the efficiency curve
    'C1': 2.3e-3,  # Coefficient for the linear term of the efficiency curve
    'C2': -1.2e-1,  # Coefficient for the constant term of the efficiency curve
    'C3': 0,  # Coefficient for constant term
    'Pnt': 15  # Night-time power consumption in watts
}

module = {
    'Name': 'AIKO Neostar 2S A500-MAH60Mb',
    'BIPV': 'N',
    'Date': '2024-10-01',
    'T_NOCT': 45,  # Nominal Operating Cell Temperature in °C
    'A_c': 2.21,  # Module area in m² (calculated from dimensions)
    'N_s': 108,  # Number of cells in series
    'I_sc_ref': 14.05,  # Reference short-circuit current in A
    'V_oc_ref': 45.02,  # Reference open-circuit voltage in V
    'I_mp_ref': 13.02,  # Reference current at maximum power point in A
    'V_mp_ref': 37.90,  # Reference voltage at maximum power point in V
    'alpha_sc': 0.00045,  # Temperature coefficient of I_sc in A/°C
    'beta_oc': -0.117052,  # Temperature coefficient of V_oc in V/°C
    'gamma_r': -0.26,  # Power temperature coefficient in %/°C
    'a_ref': 2.1,  # Modified diode ideality factor
    'I_L_ref': 14.05,  # Light-generated current in A
    'I_o_ref': 1e-10,  # Dark saturation current in A
    'R_s': 0.5,  # Series resistance in ohms
    'R_sh_ref': 300,  # Shunt resistance in ohms
    'Adjust': 0,  # Adjustment factor
    'gamma_ref': 1.1,  # Diode ideality factor
    'cells_in_series': 108,  # Number of cells in series
    'temp_ref': 25  # Reference temperature in °C
}

aiko_neostar_2s_a500 = {
    'Technology': 'mono-Si',  # Monocrystalline silicon
    'Bifacial': False,  # Not a bifacial module
    'STC': 500,  # Power output at STC in watts
     #'PTC': 475,  # Estimated PTC power in watts
    'A_c': 2.21,  # Module area in square meters (1954 mm × 1134 mm)
    'Length': 1.954,  # Module length in meters
    'Width': 1.134,  # Module width in meters
    'N_s': 120,  # Number of cells in series
    'I_sc_ref': 14.05,  # Short-circuit current at STC in amperes
    'V_oc_ref': 45.02,  # Open-circuit voltage at STC in volts
    'I_mp_ref': 13.02,  # Current at maximum power point at STC in amperes
    'V_mp_ref': 37.90,  # Voltage at maximum power point at STC in volts

    'alpha_sc': 7e-3,  # Temperature coefficient of I_sc in A/°C      ;0.05% I_sc
    'beta_oc': -0.1,  # Temperature coefficient of V_oc in V/°C    ;-0.22% V_oc
    # -0.26% Pmax
    #'T_NOCT': 45,  # Nominal Operating Cell Temperature in °C

    # CEC model parameters
    'a_ref': 1.5,  # Assumed diode ideality factor
    'I_L_ref': 14.05,  # Light-generated current at STC in amperes
    'I_o_ref': 1e-10,  # Dark saturation current at STC in amperes
    'R_s': 0.5,  # Series resistance in ohms
    'R_sh_ref': 300,  # Shunt resistance in ohms
    'Adjust': 0,  # Adjustment factor
    # 'gamma_r': -0.26,  # Temperature coefficient of power in %/°C
    # 'BIPV': False,  # Not building-integrated PV
    # 'Version': '1.0',  # Version of the module record
    # 'Date': '2024-01-01'  # Example creation date
}
"""
['Vintage' 'Area' 'Material' 'Cells_in_Series' 'Parallel_Strings' 'Isco'
 'Voco' 'Impo' 'Vmpo' 'Aisc' 'Aimp' 'C0' 'C1' 'Bvoco' 'Mbvoc' 'Bvmpo'
 'Mbvmp' 'N' 'C2' 'C3' 'A0' 'A1' 'A2' 'A3' 'A4' 'B0' 'B1' 'B2' 'B3' 'B4'
 'B5' 'DTC' 'FD' 'A' 'B' 'C4' 'C5' 'IXO' 'IXXO' 'C6' 'C7' 'Notes']
 
 
"""

temperature_model = dict(
    a=-3.47,
    b=-0.0594,
    deltaT=3
)