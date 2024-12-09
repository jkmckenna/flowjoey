#predefined_gates

from FlowCytometryTools import ThresholdGate, PolyGate

predefined_gates = {
    "u2os_lsrii_cell_gate": PolyGate([(3.558e+04, 5.712e+04), (4.842e+04, 2.892e+04), (9.184e+04, 1.452e+04), (1.515e+05, 5.292e+04), (1.538e+05, 1.015e+05), (8.957e+04, 1.333e+05), (3.558e+04, 5.712e+04)], ('FSC-A', 'SSC-A'), region='in', name='cells'),
    "u2os_lsrii_single_cell_gate": PolyGate([(4.521e+04, 3.172e+04), (5.353e+04, 2.364e+04), (1.495e+05, 8.002e+04), (1.408e+05, 8.827e+04), (4.540e+04, 3.189e+04)], ('FSC-A', 'FSC-H'), region='in', name='single_cells'),
    "u2os_lsrii_tGFP_positive_gate": ThresholdGate(2.008e+03, ('GFP'), region='above', name='GFP_positive')
}