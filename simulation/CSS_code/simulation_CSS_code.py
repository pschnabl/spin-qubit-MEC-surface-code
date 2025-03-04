import stim
import sinter
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../spin-qubit-MEC-surface-code'))
from circuits.CSS_surface_code_architecture import create_rotated_surface_code_CSS_architecture, CircuitGenParametersCSS
 
"""
Rotated surface code simulation over proposed spin qubit architecture
"""

# Generates surface code circuit tasks using Stim's circuit generation.
def generate_example_tasks(is_memory_x=False):
    etas = [100]
    probabilities = [0.0018,0.0019]
    distances = [5, 9, 13, 17]
    for eta in etas:   
        for p in probabilities:
            for d in distances:            
                rounds = 3 * d     
                params = CircuitGenParametersCSS(
                                                    rounds=rounds,
                                                    distance=d,
                                                    after_clifford1_depolarization = p/10,
                                                    before_round_data_bias_probability= (p/10, eta),
                                                    before_measure_flip_probability = 2*p,
                                                    after_reset_flip_probability = 2*p,
                                                    after_clifford2_depolarization=p,
                                                    pswap_depolarization= 0.8*p,
                                                    nswaps=(3,2), # (Ny,Nx) in the main text, defines the swaps of checks and datas (per 2 qubit gate)
                                                )
                circuit = create_rotated_surface_code_CSS_architecture(params, is_memory_x=is_memory_x)
                
                yield sinter.Task(
                    circuit=circuit,
                    decoder=None,
                    # detector_error_model=decoder_dem,
                    json_metadata={
                        'p': p,
                        'd': d,
                        "eta": eta,
                        "params": params.__dict__,
                        "memory": ["Z", "X"][is_memory_x]}       
                        )               

def main():
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "CSSRotatedThresh_23")
    # Collect the samples (takes a few minutes).
    samples = sinter.collect(
        num_workers=multiprocessing.cpu_count()-1,
        max_shots=200_000_000,
        max_errors=200000,
        tasks=[task for task in generate_example_tasks(is_memory_x=False)] + [task for task in generate_example_tasks(is_memory_x=True)],
        decoders=["pymatching"],
        #count_detection_events=True,
        save_resume_filepath= os.path.join(filepath, "results.csv")
    )
    # Print samples as CSV data.
    print(sinter.CSV_HEADER)
    for sample in samples:
        print(sample.to_csv_line())


# NOTE: This is actually necessary! If the code inside 'main()' was at the
# module level, the multiprocessing children spawned by sinter.collect would
# also attempt to run that code.
if __name__ == '__main__':
    main()