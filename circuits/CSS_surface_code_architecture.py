import stim
from typing import Set, List, Dict, ClassVar
from dataclasses import dataclass
import math
from itertools import islice
import sys
sys.path.append('../spin_qubit_architecture_circuits')
from circuits.rotated_surface_code_layout import generate_rotated_surface_code_circuit_layout

"""
Code for simulating a rotated CSS surface code over the spin qubit architecture using MEC

"""

def batched(iterable, n):
    "Batch data into tuples of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch

def append_anti_basis_error(circuit: stim.Circuit, targets: List[int], p: float, basis: str) -> None:
    if p > 0:
        if basis == "X":
            circuit.append("Z_ERROR", targets, p)
        else:
            circuit.append("X_ERROR", targets, p)

@dataclass
class CircuitGenParametersCSS():
    """
    Parameters for generating a rotated CSS surface code circuit over the architecture.
    
    Idling errors (biased):
    The probability of the idling error is scaled according the subsequent operation being done.
        parameter : before_round_data_bias_probability tuple (rate,bias)

        p = p_x + p_y + p_z and η = p_z / (p_x + p_y)
        
        therefore:
        p_x = p_y = p / [2*(1+η)]
        p_z = p*η / [1+η]
    
    Hadamard gates:
        parameter : after_clifford1_depolarization

        Single qubit depolarizing errors
        

    Two Qubit Gate errors (CNOT,CZ):
        parameter : after_clifford2_depolarization

        Two qubit depolarizing errors

    State preparation error:
        parameter : after_reset_flip_probability

        flip state to orthogonal state

    Measurement error:
        parameter : before_measure_flip_probability

        Flip measurement outcome.
    
    Swap error:
        parameter : pswap_depolarization
        parameter : nswaps

        nswap depolarizing errors in qubits that are swapped to get closer. 
    """
    
    rounds: int
    distance: int = None
    x_distance: int = None
    z_distance: int = None
    after_clifford1_depolarization: float = 0 # this will be for the single qubit gates
    before_round_data_bias_probability: tuple = (0, 0) # (p, eta) this is the idling (we relate it with the subsequent operations)
    before_measure_flip_probability: float = 0 # # this is for  the measurement errors
    after_reset_flip_probability: float = 0 # # this is for the reset errors
    exclude_other_basis_detectors: bool = False
    after_clifford2_depolarization: float = 0 # this is for the two qubit gates
    pswap_depolarization: float = 0 # error for swapping the qubits
    nswaps: int = 0 # number of swaps in proposed architecture
        
def create_rotated_surface_code_CSS_architecture(params: CircuitGenParametersCSS,
                                is_memory_x: bool = False,
                                *, 
                                exclude_other_basis_detectors: bool = False,
                                ) -> stim.Circuit:
    
    if params.rounds < 1:
        raise ValueError("Need rounds >= 1")
    if params.distance is not None and params.distance < 2:
        raise ValueError("Need a distance >= 2")    
    (x_observable,
     z_observable,
     data_coords,
     x_measure_coords,
     z_measure_coords,
     q2p,
     p2q,
     data_qubits,
     x_measurement_qubits,
     measurement_qubits,
     cnot_targets,
     measure_coord_to_order,
     data_coord_to_order,
     z_order) = generate_rotated_surface_code_circuit_layout(params.distance, params.x_distance, params.z_distance)
    z_measurement_qubits = [q for q in measurement_qubits if q not in x_measurement_qubits]

    chosen_basis_observable = x_observable if is_memory_x else z_observable
    chosen_basis_measure_coords = x_measure_coords if is_memory_x else z_measure_coords
    
    #####--CYCLE--###################################################
    # Build the repeated actions that make up the surface code cycle
    cycle_actions = stim.Circuit()
    
    # Reset the check qubits
    cycle_actions.append("TICK", []) 
    cycle_actions.append("R" + "Z", measurement_qubits)
    if params.after_reset_flip_probability > 0:
        append_anti_basis_error(cycle_actions, measurement_qubits, params.after_reset_flip_probability, "Z")

    # Biased channel on the data qubits as per idling in this time step
    if params.before_round_data_bias_probability[0] > 0:
        # We consider that for noisier operations related with longer times the idling should be higher
        idling_fact = params.after_reset_flip_probability / params.after_clifford2_depolarization
        p = idling_fact * params.before_round_data_bias_probability[0]
        eta = params.before_round_data_bias_probability[1]
        p_x = p/(2*(1+eta))
        p_y = p/(2*(1+eta))
        p_z = p*eta / (1+eta)
        cycle_actions.append("PAULI_CHANNEL_1", data_qubits, [p_x, p_y, p_z])

    # Hadamard gates to check qubits
    cycle_actions.append("TICK", []) 
    cycle_actions.append("H", measurement_qubits)
    if params.after_clifford1_depolarization > 0:
        cycle_actions.append("DEPOLARIZE1", measurement_qubits, params.after_clifford1_depolarization)
    # Biased channel on the data qubits as per idling in this time step
    if params.before_round_data_bias_probability[0] > 0:
        # We consider that for noisier operations related with longer times the idling should be higher
        idling_fact = params.after_clifford1_depolarization / params.after_clifford2_depolarization
        p = idling_fact * params.before_round_data_bias_probability[0]
        eta = params.before_round_data_bias_probability[1]
        p_x = p/(2*(1+eta))
        p_y = p/(2*(1+eta))
        p_z = p*eta / (1+eta)
        cycle_actions.append("PAULI_CHANNEL_1", data_qubits, [p_x, p_y, p_z])

    
    # Entangling gates
    for targets in cnot_targets:
        cycle_actions.append("TICK", [])
        # Add swap errors to the data and check qubits involved in the cnots, the rest will
        # idling in such step (Get them close)
        if params.pswap_depolarization > 0:
            for jj in range(params.nswaps):
                cycle_actions.append("DEPOLARIZE1", targets, params.pswap_depolarization)
                # Idling errors to unused checks while swaps occur (taylor for same as swap)
                if params.before_round_data_bias_probability[0] > 0:
                    # We consider that for noisier operations related with longer times the idling should be higher
                    idling_fact = params.pswap_depolarization / params.after_clifford2_depolarization
                    p = idling_fact * params.before_round_data_bias_probability[0]
                    eta = params.before_round_data_bias_probability[1]
                    p_x = p/(2*(1+eta))
                    p_y = p/(2*(1+eta))
                    p_z = p*eta / (1+eta)
                    # We select the data_qubits and measurement_qubits that do not belong to the targets
                    cycle_actions.append("PAULI_CHANNEL_1", list((set(data_qubits) | set(measurement_qubits)) - (set(targets))), [p_x, p_y, p_z])
        for p, pair in enumerate(batched(targets, 2)):
            # apply CZ gate to the x_measurement_qubits
            if any(q in x_measurement_qubits for q in pair):        
                cycle_actions.append("CNOT", pair)
                if params.after_clifford2_depolarization > 0:
                    cycle_actions.append("DEPOLARIZE2", pair, params.after_clifford2_depolarization)
            # apply CZ gate to the z_measurement_qubits
            else:
                cycle_actions.append("CZ", pair)
                if params.after_clifford2_depolarization > 0:
                    cycle_actions.append("DEPOLARIZE2", pair, params.after_clifford2_depolarization)
        # Idling errors to unused checks which are biased
        # our baseline error is the two qubit gate, we do not need to rescale the idling for those here
        if params.before_round_data_bias_probability[0] > 0:
            p = params.before_round_data_bias_probability[0]
            eta = params.before_round_data_bias_probability[1]
            p_x = p/(2*(1+eta))
            p_y = p/(2*(1+eta))
            p_z = p*eta / (1+eta)
            # We select the data_qubits and measurement_qubits that do not belong to the targets
            cycle_actions.append("PAULI_CHANNEL_1", list((set(data_qubits) | set(measurement_qubits)) - (set(targets))), [p_x, p_y, p_z])
        # Add swap errors to the data and check qubits involved in the cnots, the rest will
        # idling in such step (back to original position)
        if params.pswap_depolarization > 0:
            for jj in range(params.nswaps):
                cycle_actions.append("DEPOLARIZE1", targets, params.pswap_depolarization)
                # Idling errors to unused checks while swaps occur (taylor for same as swap)
                if params.before_round_data_bias_probability[0] > 0:
                    # We consider that for noisier operations related with longer times the idling should be higher
                    idling_fact = params.pswap_depolarization / params.after_clifford2_depolarization
                    p = idling_fact * params.before_round_data_bias_probability[0]
                    eta = params.before_round_data_bias_probability[1]
                    p_x = p/(2*(1+eta))
                    p_y = p/(2*(1+eta))
                    p_z = p*eta / (1+eta)
                    # We select the data_qubits and measurement_qubits that do not belong to the targets
                    cycle_actions.append("PAULI_CHANNEL_1", list((set(data_qubits) | set(measurement_qubits)) - (set(targets))), [p_x, p_y, p_z])

    # Hadamard gates to check qubits
    cycle_actions.append("TICK", [])    
    cycle_actions.append("H", measurement_qubits)
    if params.after_clifford1_depolarization > 0:
        cycle_actions.append("DEPOLARIZE1", measurement_qubits, params.after_clifford1_depolarization)
    # Biased channel on the data qubits as per idling in this time step
    if params.before_round_data_bias_probability[0] > 0:
        # rescale idling as 1 qubit gate
        idling_fact = params.after_clifford1_depolarization / params.after_clifford2_depolarization
        p = idling_fact * params.before_round_data_bias_probability[0]
        eta = params.before_round_data_bias_probability[1]
        p_x = p/(2*(1+eta))
        p_y = p/(2*(1+eta))
        p_z = p*eta / (1+eta)
        cycle_actions.append("PAULI_CHANNEL_1", data_qubits, [p_x, p_y, p_z])
    
    # Measure the check qubits
    cycle_actions.append("TICK", [])    
    if params.before_measure_flip_probability > 0:
        append_anti_basis_error(cycle_actions, measurement_qubits, params.before_measure_flip_probability, basis="Z")
    cycle_actions.append("M" + "Z", measurement_qubits)

    # Biased channel on the data qubits as per idling in this time step
    if params.before_round_data_bias_probability[0] > 0:
        # rescale idling as measurement
        idling_fact = params.before_measure_flip_probability / params.after_clifford2_depolarization
        p = idling_fact * params.before_round_data_bias_probability[0]
        eta = params.before_round_data_bias_probability[1]
        p_x = p/(2*(1+eta))
        p_y = p/(2*(1+eta))
        p_z = p*eta / (1+eta)
        cycle_actions.append("PAULI_CHANNEL_1", data_qubits, [p_x, p_y, p_z])


    ####--HEAD--####################################################
    # Build the start of the circuit, getting a state that's ready to cycle
    # In particular, the first cycle has different detectors and so has to be handled special.
    head = stim.Circuit()
    for k, v in sorted(q2p.items()):
        head.append("QUBIT_COORDS", [k], [v.real, v.imag])
    
    # Reset the data qubits
    head.append("TICK", []) 
    head.append("R" + "ZX"[is_memory_x], data_qubits)
    if params.after_reset_flip_probability > 0:
        append_anti_basis_error(head, data_qubits, params.after_reset_flip_probability, "ZX"[is_memory_x])
    
    # We have compiled the rotated planar code with CZ for the Z checks and every Hadamard gate in state prep
    # has been made to form a |+> state for the check, here explicitly with the Hadamard gate
    head.append("R" + "Z", measurement_qubits)
    if params.after_reset_flip_probability > 0:
        append_anti_basis_error(head, measurement_qubits, params.after_reset_flip_probability, "Z")

    # Hadamard gates to check qubits
    head.append("TICK", [])     
    head.append("H", measurement_qubits)
    if params.after_clifford1_depolarization > 0:
        head.append("DEPOLARIZE1", measurement_qubits, params.after_clifford1_depolarization)
    # Biased channel on the data qubits as per idling in this time step
    if params.before_round_data_bias_probability[0] > 0:
        idling_fact = params.after_clifford1_depolarization / params.after_clifford2_depolarization
        p = idling_fact * params.before_round_data_bias_probability[0]
        eta = params.before_round_data_bias_probability[1]
        p_x = p/(2*(1+eta))
        p_y = p/(2*(1+eta))
        p_z = p*eta / (1+eta)
        head.append("PAULI_CHANNEL_1", data_qubits, [p_x, p_y, p_z])
    
    # Apply entangling gates
    for targets in cnot_targets:
        head.append("TICK", [])
        # Add swap errors to the data and check qubits involved in the cnots, the rest will
        # idling in such step (Get them close)
        if params.pswap_depolarization > 0:
            for jj in range(params.nswaps):
                head.append("DEPOLARIZE1", targets, params.pswap_depolarization)
                # Idling errors to unused checks while swaps occur (taylor for same as swap)
                if params.before_round_data_bias_probability[0] > 0:
                    # We consider that for noisier operations related with longer times the idling should be higher
                    idling_fact = params.pswap_depolarization / params.after_clifford2_depolarization
                    p = idling_fact * params.before_round_data_bias_probability[0]
                    eta = params.before_round_data_bias_probability[1]
                    p_x = p/(2*(1+eta))
                    p_y = p/(2*(1+eta))
                    p_z = p*eta / (1+eta)
                    # We select the data_qubits and measurement_qubits that do not belong to the targets
                    head.append("PAULI_CHANNEL_1", list((set(data_qubits) | set(measurement_qubits)) - (set(targets))), [p_x, p_y, p_z])
        for p, pair in enumerate(batched(targets, 2)):
            # apply CZ gate to the x_measurement_qubits
            if any(q in x_measurement_qubits for q in pair):        
                head.append("CNOT", pair)
                if params.after_clifford2_depolarization > 0:
                    head.append("DEPOLARIZE2", pair, params.after_clifford2_depolarization)
            # apply CZ gate to the z_measurement_qubits
            else:
                head.append("CZ", pair)
                if params.after_clifford2_depolarization > 0:
                    head.append("DEPOLARIZE2", pair, params.after_clifford2_depolarization)
        # Idling errors to unused checks which are biased
        #our baseline error rate is the 2 qubit gate, no need to rescale here
        if params.before_round_data_bias_probability[0] > 0:
            p = params.before_round_data_bias_probability[0]
            eta = params.before_round_data_bias_probability[1]
            p_x = p/(2*(1+eta))
            p_y = p/(2*(1+eta))
            p_z = p*eta / (1+eta)
            # We select the data_qubits and measurement_qubits that do not belong to the targets
            head.append("PAULI_CHANNEL_1", list((set(data_qubits) | set(measurement_qubits)) - (set(targets))), [p_x, p_y, p_z])
        # Add swap errors to the data and check qubits involved in the cnots, the rest will
        # idling in such step (Get them back to OG position)
        if params.pswap_depolarization > 0:
            for jj in range(params.nswaps):
                head.append("DEPOLARIZE1", targets, params.pswap_depolarization)
                # Idling errors to unused checks while swaps occur (taylor for same as swap)
                if params.before_round_data_bias_probability[0] > 0:
                    # We consider that for noisier operations related with longer times the idling should be higher
                    idling_fact = params.pswap_depolarization / params.after_clifford2_depolarization
                    p = idling_fact * params.before_round_data_bias_probability[0]
                    eta = params.before_round_data_bias_probability[1]
                    p_x = p/(2*(1+eta))
                    p_y = p/(2*(1+eta))
                    p_z = p*eta / (1+eta)
                    # We select the data_qubits and measurement_qubits that do not belong to the targets
                    head.append("PAULI_CHANNEL_1", list((set(data_qubits) | set(measurement_qubits)) - (set(targets))), [p_x, p_y, p_z])
    
    # Hadamard gates to check qubits
    head.append("TICK", [])    
    head.append("H", measurement_qubits)
    if params.after_clifford1_depolarization > 0:
        head.append("DEPOLARIZE1", measurement_qubits, params.after_clifford1_depolarization)
    # Biased channel on the data qubits as per idling in this time step
    if params.before_round_data_bias_probability[0] > 0:
        idling_fact = params.after_clifford1_depolarization / params.after_clifford2_depolarization
        p = idling_fact * params.before_round_data_bias_probability[0]
        eta = params.before_round_data_bias_probability[1]
        p_x = p/(2*(1+eta))
        p_y = p/(2*(1+eta))
        p_z = p*eta / (1+eta)
        head.append("PAULI_CHANNEL_1", data_qubits, [p_x, p_y, p_z])


    # Measure the check qubits
    head.append("TICK", [])    
    if params.before_measure_flip_probability > 0:
        append_anti_basis_error(head, measurement_qubits, params.before_measure_flip_probability, basis="Z")
    head.append("M" + "Z", measurement_qubits)

    # Biased channel on the data qubits as per idling in this time step
    if params.before_round_data_bias_probability[0] > 0:
        idling_fact = params.before_measure_flip_probability / params.after_clifford2_depolarization
        p = idling_fact * params.before_round_data_bias_probability[0]
        eta = params.before_round_data_bias_probability[1]
        p_x = p/(2*(1+eta))
        p_y = p/(2*(1+eta))
        p_z = p*eta / (1+eta)
        head.append("PAULI_CHANNEL_1", data_qubits, [p_x, p_y, p_z])

    # head += cycle_actions
    for measure in sorted(chosen_basis_measure_coords, key=lambda c: (c.real, c.imag)):
        head.append(
            "DETECTOR",
            [stim.target_rec(-len(measurement_qubits) + measure_coord_to_order[measure])],
            [measure.real, measure.imag, 0.0]
        )

    ####--BODY--####################################################
    # Build the repeated body of the circuit, including the detectors comparing to previous cycles.
    body = cycle_actions.copy()
    m = len(measurement_qubits)
    body.append("SHIFT_COORDS", [], [0.0, 0.0, 1.0])
    for m_index in measurement_qubits:
        m_coord = q2p[m_index]
        k = len(measurement_qubits) - measure_coord_to_order[m_coord] - 1
        if not exclude_other_basis_detectors or m_coord in chosen_basis_measure_coords:
            body.append(
                "DETECTOR",
                [stim.target_rec(-k - 1), stim.target_rec(-k - 1 - m)],
                [m_coord.real, m_coord.imag, 0.0]
            )
    

    ####--TAIL--####################################################
    # Build the end of the circuit, getting out of the cycle state and terminating.
    # In particular, the data measurements create detectors that have to be handled special.
    # Also, the tail is responsible for identifying the logical observable.
    tail = stim.Circuit()
     # Reset the checks
    tail.append("TICK", []) 
    tail.append("R" + "Z", measurement_qubits)
    if params.after_reset_flip_probability > 0:
        append_anti_basis_error(tail, measurement_qubits, params.after_reset_flip_probability, "Z")
    # Biased channel on the data qubits as per idling in this time step
    if params.before_round_data_bias_probability[0] > 0:
        idling_fact = params.after_reset_flip_probability / params.after_clifford2_depolarization
        p = idling_fact * params.before_round_data_bias_probability[0]
        eta = params.before_round_data_bias_probability[1]
        p_x = p/(2*(1+eta))
        p_y = p/(2*(1+eta))
        p_z = p*eta / (1+eta)
        tail.append("PAULI_CHANNEL_1", data_qubits, [p_x, p_y, p_z])

    # Hadamard gates to check qubits
    tail.append("TICK", [])     
    tail.append("H", measurement_qubits)
    if params.after_clifford1_depolarization > 0:
        tail.append("DEPOLARIZE1", measurement_qubits, params.after_clifford1_depolarization)
    # Biased channel on the data qubits as per idling in this time step
    if params.before_round_data_bias_probability[0] > 0:
        idling_fact = params.after_clifford1_depolarization / params.after_clifford2_depolarization
        p = idling_fact * params.before_round_data_bias_probability[0]
        eta = params.before_round_data_bias_probability[1]
        p_x = p/(2*(1+eta))
        p_y = p/(2*(1+eta))
        p_z = p*eta / (1+eta)
        tail.append("PAULI_CHANNEL_1", data_qubits, [p_x, p_y, p_z])

    # Apply entangling gates
    
    for targets in cnot_targets:
        tail.append("TICK", [])
        # Add swap errors to the data and check qubits involved in the cnots, the rest will
        # idling in such step (Get them close)
        if params.pswap_depolarization > 0:
            for jj in range(params.nswaps):
                tail.append("DEPOLARIZE1", targets, params.pswap_depolarization)
                # Idling errors to unused checks while swaps occur (taylor for same as swap)
                if params.before_round_data_bias_probability[0] > 0:
                    # We consider that for noisier operations related with longer times the idling should be higher
                    idling_fact = params.pswap_depolarization / params.after_clifford2_depolarization
                    p = idling_fact * params.before_round_data_bias_probability[0]
                    eta = params.before_round_data_bias_probability[1]
                    p_x = p/(2*(1+eta))
                    p_y = p/(2*(1+eta))
                    p_z = p*eta / (1+eta)
                    # We select the data_qubits and measurement_qubits that do not belong to the targets
                    tail.append("PAULI_CHANNEL_1", list((set(data_qubits) | set(measurement_qubits)) - (set(targets))), [p_x, p_y, p_z])
        for p, pair in enumerate(batched(targets, 2)):
            # apply CZ gate to the x_measurement_qubits
            if any(q in x_measurement_qubits for q in pair):        
                tail.append("CNOT", pair)
                if params.after_clifford2_depolarization > 0:
                    tail.append("DEPOLARIZE2", pair, params.after_clifford2_depolarization)
            # apply CZ gate to the z_measurement_qubits
            else:
                tail.append("CZ", pair)
                if params.after_clifford2_depolarization > 0:
                    tail.append("DEPOLARIZE2", pair, params.after_clifford2_depolarization)
        # Idling errors to unused checks which are biased
        # our baseline error rate is the 2 qubit one, not need to rescale here
        if params.before_round_data_bias_probability[0] > 0:
            p = params.before_round_data_bias_probability[0]
            eta = params.before_round_data_bias_probability[1]
            p_x = p/(2*(1+eta))
            p_y = p/(2*(1+eta))
            p_z = p*eta / (1+eta)
            # We select the data_qubits and measurement_qubits that do not belong to the targets
            tail.append("PAULI_CHANNEL_1", list((set(data_qubits) | set(measurement_qubits)) - (set(targets))), [p_x, p_y, p_z])
        # Add swap errors to the data and check qubits involved in the cnots, the rest will
        # idling in such step (Get them back to OG position)
        if params.pswap_depolarization > 0:
            for jj in range(params.nswaps):
                tail.append("DEPOLARIZE1", targets, params.pswap_depolarization)
                # Idling errors to unused checks while swaps occur (taylor for same as swap)
                if params.before_round_data_bias_probability[0] > 0:
                    # We consider that for noisier operations related with longer times the idling should be higher
                    idling_fact = params.pswap_depolarization / params.after_clifford2_depolarization
                    p = idling_fact * params.before_round_data_bias_probability[0]
                    eta = params.before_round_data_bias_probability[1]
                    p_x = p/(2*(1+eta))
                    p_y = p/(2*(1+eta))
                    p_z = p*eta / (1+eta)
                    # We select the data_qubits and measurement_qubits that do not belong to the targets
                    tail.append("PAULI_CHANNEL_1", list((set(data_qubits) | set(measurement_qubits)) - (set(targets))), [p_x, p_y, p_z])
    
    # Hadamard gates to check qubits
    tail.append("TICK", [])   
    tail.append("H", measurement_qubits)
    if params.after_clifford1_depolarization > 0:
        tail.append("DEPOLARIZE1", measurement_qubits, params.after_clifford1_depolarization)
    # Biased channel on the data qubits as per idling in this time step
    if params.before_round_data_bias_probability[0] > 0:
        idling_fact = params.after_clifford1_depolarization / params.after_clifford2_depolarization
        p = idling_fact * params.before_round_data_bias_probability[0]
        eta = params.before_round_data_bias_probability[1]
        p_x = p/(2*(1+eta))
        p_y = p/(2*(1+eta))
        p_z = p*eta / (1+eta)
        tail.append("PAULI_CHANNEL_1", data_qubits, [p_x, p_y, p_z])

    # Measure the check qubits
    tail.append("TICK", [])    
    if params.before_measure_flip_probability > 0:
        append_anti_basis_error(cycle_actions, measurement_qubits, params.before_measure_flip_probability, basis="Z")
    tail.append("M" + "Z", measurement_qubits)

    
    # Detectors of checks last round
    m = len(measurement_qubits)
    body.append("SHIFT_COORDS", [], [0.0, 0.0, 1.0])
    for m_index in measurement_qubits:
        m_coord = q2p[m_index]
        k = len(measurement_qubits) - measure_coord_to_order[m_coord] - 1
        if not exclude_other_basis_detectors or m_coord in chosen_basis_measure_coords:
            tail.append(
                "DETECTOR",
                [stim.target_rec(-k - 1), stim.target_rec(-k - 1 - m)],
                [m_coord.real, m_coord.imag, 0.0]
            )

    # Measure the data qubits
    if params.before_measure_flip_probability > 0:
        append_anti_basis_error(tail, data_qubits, params.before_measure_flip_probability, "ZX"[is_memory_x])
    tail.append("M" + "ZX"[is_memory_x], data_qubits)
    
    # Detectors
    for measure in sorted(chosen_basis_measure_coords, key=lambda c: (c.real, c.imag)):
        detectors: List[int] = []
        for delta in z_order:
            data = measure + delta
            if data in p2q:
                detectors.append(-len(data_qubits) + data_coord_to_order[data])
        detectors.append(-len(data_qubits) - len(measurement_qubits) + measure_coord_to_order[measure])
        detectors.sort(reverse=True)
        tail.append("DETECTOR", [stim.target_rec(x) for x in detectors], [measure.real, measure.imag, 1.0])

    # Logical observable
    obs_inc: List[int] = []
    for q in chosen_basis_observable:
        obs_inc.append(-len(data_qubits) + data_coord_to_order[q])
    obs_inc.sort(reverse=True)
    tail.append("OBSERVABLE_INCLUDE", [stim.target_rec(x) for x in obs_inc], 0.0)

    # Combine to form final circuit.
    return head + body * (params.rounds - 2) + tail