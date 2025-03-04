import stim
from typing import Set, List, Dict, ClassVar
from dataclasses import dataclass
import sys
sys.path.append('../spin-qubit-MEC-surface-code')
from circuits.rotated_surface_code_layout import generate_rotated_surface_code_circuit_layout

from typing import Set, List, Dict
import math

"""
Code for simulating a rotated XZZX surface code over the spin qubit architecture using MEC
"""

def append_anti_basis_error(circuit: stim.Circuit, targets: List[int], p: float, basis: str) -> None:
    if p > 0:
        if basis == "X":
            circuit.append("Z_ERROR", targets, p) # I have now removed the 2/3
        else:
            circuit.append("X_ERROR", targets, p)


def generate_XZZX_circuit_layout(distance: int):
    """
    Generate the layout of a rotated XZZX surface code with the given distance.
    
    This is essentially the same as the layout of the rotated surface code,
    but with changed orders (z_order, x_order) of applying the CNOT and
    CZ gates.`cnot_targets` is a list of 4 lists defining the 4 steps in which the
    CNOT and CZ gates are applied. The first and last lists correspond to CNOT gates,
    the second and third lists correspond to CZ gates.

    Args:
        distance (int): Distance of the surface code.

    Returns:
        Tuple: A tuple containing the following elements (x_observable, z_observable, data_coords, x_measure_coords, z_measure_coords, q2p, p2q, data_qubits, x_measurement_qubits, measurement_qubits, cnot_targets, measure_coord_to_order, data_coord_to_order, z_order):
            - x_observable (List[complex]): List of complex numbers representing the x-observable qubits.
            - z_observable (List[complex]): List of complex numbers representing the z-observable qubits.
            - data_coords (Set[complex]): Set of complex numbers representing the data qubits.
            - x_measure_coords (Set[complex]): Set of complex numbers representing the x-measurement qubits.
            - z_measure_coords (Set[complex]): Set of complex numbers representing the z-measurement qubits.
            - q2p (Dict[int, complex]): Dictionary mapping qubit indices to complex numbers indicating their coordinates.
            - p2q (Dict[complex, int]): Dictionary mapping complex numbers to qubit indices.
            - data_qubits (List[int]): List of qubit indices representing the data qubits.
            - x_measurement_qubits (List[int]): List of qubit indices representing the x-measurement qubits.
            - measurement_qubits (List[int]): List of qubit indices representing the measurement qubits.
            - cnot_targets (List[List[int]]): List of lists of qubit indices representing the CNOT and CZ gate targets.
            - measure_coord_to_order (Dict[complex, int]): Dictionary mapping complex numbers to integers indicating the order of the measurement qubits.
            - data_coord_to_order (Dict[complex, int]): Dictionary mapping complex numbers to integers indicating the order of the data qubits.
            - z_order (List[complex]): List of complex numbers representing the z-order of the qubits.
    """
    # x and z distance treated similarly for now. Can be changed later.
    x_distance = distance
    z_distance = distance

    # Place data qubits
    data_coords: Set[complex] = set()
    x_observable: List[complex] = []
    z_observable: List[complex] = []
    for x in [i + 0.5 for i in range(z_distance)]:
        for y in [i + 0.5 for i in range(x_distance)]:
            q = x * 2 + y * 2 * 1j
            data_coords.add(q)
            if y == 0.5:
                z_observable.append(q)
            if x == 0.5:
                x_observable.append(q)

    # Place measurement qubits.
    x_measure_coords: Set[complex] = set()
    z_measure_coords: Set[complex] = set()
    for x in range(z_distance + 1):
        for y in range(x_distance + 1):
            q = x * 2 + y * 2j
            on_boundary_1 = x == 0 or x == z_distance
            on_boundary_2 = y == 0 or y == x_distance
            parity = (x % 2) != (y % 2)
            if on_boundary_1 and parity:
                continue
            if on_boundary_2 and not parity:
                continue
            if parity:
                x_measure_coords.add(q)
            else:
                z_measure_coords.add(q)

    def coord_to_idx(q: complex) -> int:
        q = q - math.fmod(q.real, 2) * 1j
        return int(q.real + q.imag * (z_distance + 0.5))

    # Index the measurement qubits and data qubits.
    p2q: Dict[complex, int] = {}
    for q in data_coords:
        p2q[q] = coord_to_idx(q)

    for q in x_measure_coords:
        p2q[q] = coord_to_idx(q)

    for q in z_measure_coords:
        p2q[q] = coord_to_idx(q)

    q2p: Dict[int, complex] = {v: k for k, v in p2q.items()}

    data_qubits = [p2q[q] for q in data_coords]
    measurement_qubits = [p2q[q] for q in x_measure_coords]
    measurement_qubits += [p2q[q] for q in z_measure_coords]
    x_measurement_qubits = [p2q[q] for q in x_measure_coords]

    all_qubits: List[int] = []
    all_qubits += data_qubits + measurement_qubits

    all_qubits.sort()
    data_qubits.sort()
    measurement_qubits.sort()
    x_measurement_qubits.sort()

    # Reverse index the measurement order used for defining detectors
    data_coord_to_order: Dict[complex, int] = {}
    measure_coord_to_order: Dict[complex, int] = {}
    for q in data_qubits:
        data_coord_to_order[q2p[q]] = len(data_coord_to_order)
    for q in measurement_qubits:
        measure_coord_to_order[q2p[q]] = len(measure_coord_to_order)

    # Define interaction orders so that hook errors run against the error grain instead of with it.
    z_order: List[complex] = [-1 - 1j, -1 + 1j, +1 - 1j, +1 + 1j] 
    x_order: List[complex] = [-1 - 1j, 1 - 1j, -1 + 1j, 1 + 1j]
    
    # List out CNOT and CZ gate targets using given interaction orders.
    cnot_targets: List[List[int]] = [[], [], [], []]
    for k in range(4):
        for measure in sorted(x_measure_coords, key=lambda c: (c.real, c.imag)):
            data = measure + x_order[k]
            if data in p2q:
                cnot_targets[k].append(p2q[measure])
                cnot_targets[k].append(p2q[data])

        for measure in sorted(z_measure_coords, key=lambda c: (c.real, c.imag)):
            data = measure + z_order[k]
            if data in p2q:
                cnot_targets[k].append(p2q[measure])
                cnot_targets[k].append(p2q[data])
                
    return (x_observable,
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
            z_order)



@dataclass
class CircuitGenParametersXZZX():
    """
    Parameters for generating a rotated XZZX surface code circuit over the architecture.
    
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
    after_clifford1_depolarization: float = 0 # this will be for the single qubit gates
    before_round_data_bias_probability: tuple = (0, 0) # (p, eta) this is the idling (we relate it with the subsequent operations)
    before_measure_flip_probability: float = 0 # this is for  the measurement errors
    after_reset_flip_probability: float = 0 # this is for the reset errors
    after_clifford2_depolarization: float = 0 # this is for the two qubit gates
    pswap_depolarization: float = 0 # error for swapping the qubits
    nswaps: tuple = (0, 0) # (Ny,Nx) in the main text, defines the swaps of checks and datas (per 2 qubit gate)
    
def create_rotated_XZZX_surface_code_architecture(params: CircuitGenParametersXZZX,
                                                  is_memory_H: bool = False,
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
     z_order) = generate_XZZX_circuit_layout(params.distance)
    
    # define which data qubits are reset in the X basis and the which in the Z basis.
    # This is equivalent to applying a Hadarmad gate on every other data qubit. There are two possibilities:
    if is_memory_H: 
        data_qubits_x = data_qubits[::2]
        data_qubits_z = data_qubits[1::2]
    else:
        data_qubits_x = data_qubits[1::2]
        data_qubits_z = data_qubits[::2]
    

    # x_observable corresponds to the left vertical line of data qubits
    # z_observable corresponds to the bottom horizontal line of data qubits
    chosen_basis_observable = z_observable if is_memory_H else x_observable
    chosen_basis_measure_coords = z_measure_coords if is_memory_H else x_measure_coords

    #####--CYCLE--###################################################
    # Build the repeated actions that make up the surface code cycle
    cycle_actions = stim.Circuit()
    
    cycle_actions.append("TICK", [])   
    # Reset the check qubits
    cycle_actions.append("RZ", measurement_qubits)
    if params.after_reset_flip_probability > 0:
        append_anti_basis_error(cycle_actions, measurement_qubits, params.after_reset_flip_probability, basis="Z")
        cycle_actions.append("TICK", []) 
    
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


    # the two qubit gates for extracting the XZZX stabilizers are applied in 4 steps
    # 1. apply cnots
    cycle_actions.append("TICK", [])
    # Add swap errors to the data and check qubits involved in the cnots, the rest will
    # idling in such step (Get them close)
    if params.pswap_depolarization > 0:
        for jj in range(params.nswaps[0]):
            cycle_actions.append("DEPOLARIZE1", list(set(cnot_targets[0]) & set(measurement_qubits)) , params.pswap_depolarization)
        for jj in range(params.nswaps[1]):
            cycle_actions.append("DEPOLARIZE1", list(set(cnot_targets[0]) & set(data_qubits)) , params.pswap_depolarization)
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
                cycle_actions.append("PAULI_CHANNEL_1", list((set(data_qubits) | set(measurement_qubits)) - (set(cnot_targets[0]))), [p_x, p_y, p_z])
    cycle_actions.append("CNOT", cnot_targets[0])
    if params.after_clifford2_depolarization > 0:
        cycle_actions.append("DEPOLARIZE2", cnot_targets[0], params.after_clifford2_depolarization)
    # Idling errors to unused checks (while CNOTs operating)
    # our baseline error is the two qubit gate, we do not need to rescale the idling for those here
    if params.before_round_data_bias_probability[0] > 0:
        p = params.before_round_data_bias_probability[0]
        eta = params.before_round_data_bias_probability[1]
        p_x = p/(2*(1+eta))
        p_y = p/(2*(1+eta))
        p_z = p*eta / (1+eta)
        # We select the data_qubits and measurement_qubits that do not belong to the targets
        cycle_actions.append("PAULI_CHANNEL_1", list((set(data_qubits) | set(measurement_qubits)) - (set(cnot_targets[0]))), [p_x, p_y, p_z])
    # Add swap errors to the data and check qubits involved in the cnots, the rest will
    # idling in such step (go back to original position)
    if params.pswap_depolarization > 0:
        for jj in range(params.nswaps[0]):
            cycle_actions.append("DEPOLARIZE1", list(set(cnot_targets[0]) & set(measurement_qubits)) , params.pswap_depolarization)
        for jj in range(params.nswaps[1]):
            cycle_actions.append("DEPOLARIZE1", list(set(cnot_targets[0]) & set(data_qubits)) , params.pswap_depolarization)
            # Idling errors to unused checks while swaps occur (taylor for same as swap)
            if params.before_round_data_bias_probability[0] > 0:
                idling_fact = params.pswap_depolarization / params.after_clifford2_depolarization
                p = idling_fact * params.before_round_data_bias_probability[0]
                eta = params.before_round_data_bias_probability[1]
                p_x = p/(2*(1+eta))
                p_y = p/(2*(1+eta))
                p_z = p*eta / (1+eta)
                # We select the data_qubits and measurement_qubits that do not belong to the targets
                cycle_actions.append("PAULI_CHANNEL_1", list((set(data_qubits) | set(measurement_qubits)) - (set(cnot_targets[0]))), [p_x, p_y, p_z])
    
    # 2. & 3. apply CZ gates
    for targets in cnot_targets[1:3]:
        cycle_actions.append("TICK", [])
        # Add swap errors to the data and check qubits involved in the cnots, the rest will
        # idling in such step (Get them close)
        if params.pswap_depolarization > 0:
            for jj in range(params.nswaps[0]):
                cycle_actions.append("DEPOLARIZE1", list(set(targets) & set(measurement_qubits)) , params.pswap_depolarization)
            for jj in range(params.nswaps[1]):
                cycle_actions.append("DEPOLARIZE1", list(set(targets) & set(data_qubits)) , params.pswap_depolarization)
                # Idling errors to unused checks while swaps occur (taylor for same as swap)
                if params.before_round_data_bias_probability[0] > 0:
                    idling_fact = params.pswap_depolarization / params.after_clifford2_depolarization
                    p = idling_fact * params.before_round_data_bias_probability[0]
                    eta = params.before_round_data_bias_probability[1]
                    p_x = p/(2*(1+eta))
                    p_y = p/(2*(1+eta))
                    p_z = p*eta / (1+eta)
                    # We select the data_qubits and measurement_qubits that do not belong to the targets
                    cycle_actions.append("PAULI_CHANNEL_1", list((set(data_qubits) | set(measurement_qubits)) - (set(targets))), [p_x, p_y, p_z])
        cycle_actions.append("CZ", targets)
        if params.after_clifford2_depolarization > 0:
            cycle_actions.append("DEPOLARIZE2", targets, params.after_clifford2_depolarization)
        # Idling errors to unused checks which are biased (while CZs operating)
        # our baseline error rate is the two qubit gate, we do not need to rescale the idling here
        if params.before_round_data_bias_probability[0] > 0:
            p = params.before_round_data_bias_probability[0]
            p_x = p/(2*(1+eta))
            p_y = p/(2*(1+eta))
            p_z = p*eta / (1+eta)
            # We select the data_qubits and measurement_qubits that do not belong to the targets
            cycle_actions.append("PAULI_CHANNEL_1", list((set(data_qubits) | set(measurement_qubits)) - (set(targets))), [p_x, p_y, p_z])
        # Add swap errors to the data and check qubits involved in the cnots, the rest will
        # idling in such step (go back to original position)
        if params.pswap_depolarization > 0:
            for jj in range(params.nswaps[0]):
                cycle_actions.append("DEPOLARIZE1", list(set(targets) & set(measurement_qubits)) , params.pswap_depolarization)
            for jj in range(params.nswaps[1]):
                cycle_actions.append("DEPOLARIZE1", list(set(targets) & set(data_qubits)) , params.pswap_depolarization)
                # Idling errors to unused checks while swaps occur (taylor for same as swap)
                if params.before_round_data_bias_probability[0] > 0:
                    idling_fact = params.pswap_depolarization / params.after_clifford2_depolarization
                    p = idling_fact * params.before_round_data_bias_probability[0]
                    eta = params.before_round_data_bias_probability[1]
                    p_x = p/(2*(1+eta))
                    p_y = p/(2*(1+eta))
                    p_z = p*eta / (1+eta)
                    # We select the data_qubits and measurement_qubits that do not belong to the targets
                    cycle_actions.append("PAULI_CHANNEL_1", list((set(data_qubits) | set(measurement_qubits)) - (set(targets))), [p_x, p_y, p_z])
    # 4. apply cnots
    cycle_actions.append("TICK", [])
    # Add swap errors to the data and check qubits involved in the cnots, the rest will
    # idling in such step
    if params.pswap_depolarization > 0:
        for jj in range(params.nswaps[0]):
            cycle_actions.append("DEPOLARIZE1", list(set(cnot_targets[3]) & set(measurement_qubits)) , params.pswap_depolarization)
        for jj in range(params.nswaps[1]):
            cycle_actions.append("DEPOLARIZE1", list(set(cnot_targets[3]) & set(data_qubits)) , params.pswap_depolarization)
            # Idling errors to unused checks while swaps occur (taylor for same as swap)
            if params.before_round_data_bias_probability[0] > 0:
                idling_fact = params.pswap_depolarization / params.after_clifford2_depolarization
                p = idling_fact * params.before_round_data_bias_probability[0]
                eta = params.before_round_data_bias_probability[1]
                p_x = p/(2*(1+eta))
                p_y = p/(2*(1+eta))
                p_z = p*eta / (1+eta)
                # We select the data_qubits and measurement_qubits that do not belong to the targets
                cycle_actions.append("PAULI_CHANNEL_1", list((set(data_qubits) | set(measurement_qubits)) - (set(cnot_targets[3]))), [p_x, p_y, p_z])
    cycle_actions.append("CNOT", cnot_targets[3])
    if params.after_clifford2_depolarization > 0:
        cycle_actions.append("DEPOLARIZE2", cnot_targets[3], params.after_clifford2_depolarization)
    # Idling errors to unused checks (while CNOTs operating)
    # our baseline error is the two qubit gate, we do not need to rescale the idling here
    if params.before_round_data_bias_probability[0] > 0:
        p = params.before_round_data_bias_probability[0]
        eta = params.before_round_data_bias_probability[1]
        p_x = p/(2*(1+eta))
        p_y = p/(2*(1+eta))
        p_z = p*eta / (1+eta)
        # We select the data_qubits and measurement_qubits that do not belong to the targets
        cycle_actions.append("PAULI_CHANNEL_1", list((set(data_qubits) | set(measurement_qubits)) - (set(cnot_targets[3]))), [p_x, p_y, p_z])
    # Add swap errors to the data and check qubits involved in the cnots, the rest will
    # idling in such step (go back to original position)
    if params.pswap_depolarization > 0:
        for jj in range(params.nswaps[0]):
            cycle_actions.append("DEPOLARIZE1", list(set(cnot_targets[3]) & set(measurement_qubits)) , params.pswap_depolarization)
        for jj in range(params.nswaps[1]):
            cycle_actions.append("DEPOLARIZE1", list(set(cnot_targets[3]) & set(data_qubits)) , params.pswap_depolarization)
            # Idling errors to unused checks while swaps occur (taylor for same as swap)
            if params.before_round_data_bias_probability[0] > 0:
                idling_fact = params.pswap_depolarization / params.after_clifford2_depolarization
                p = idling_fact * params.before_round_data_bias_probability[0]
                eta = params.before_round_data_bias_probability[1]
                p_x = p/(2*(1+eta))
                p_y = p/(2*(1+eta))
                p_z = p*eta / (1+eta)
                # We select the data_qubits and measurement_qubits that do not belong to the targets
                cycle_actions.append("PAULI_CHANNEL_1", list((set(data_qubits) | set(measurement_qubits)) - (set(cnot_targets[3]))), [p_x, p_y, p_z])

    # Apply Hadamard gates to checks
    cycle_actions.append("TICK", [])      
    cycle_actions.append("H", measurement_qubits)
    if params.after_clifford1_depolarization > 0:
        cycle_actions.append("DEPOLARIZE1", measurement_qubits, params.after_clifford1_depolarization)
    # Biased channel on the data qubits as per idling in this time step
    if params.before_round_data_bias_probability[0] > 0:
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
        idling_fact = params.before_measure_flip_probability / params.after_clifford2_depolarization
        p = idling_fact * params.before_round_data_bias_probability[0]
        eta = params.before_round_data_bias_probability[1]
        p_x = p/(2*(1+eta))
        p_y = p/(2*(1+eta))
        p_z = p*eta / (1+eta)
        cycle_actions.append("PAULI_CHANNEL_1", data_qubits, [p_x, p_y, p_z])
  
    #####--HEAD--####################################################  
    head = stim.Circuit()
    for k, v in sorted(q2p.items()):
        head.append("QUBIT_COORDS", [k], [v.real, v.imag])
    
    head.append("TICK", [])
    # Reset the data qubits: every other qubit is reset to |0> and the rest to |+>
    head.append("RX", data_qubits_x)
    head.append("R", data_qubits_z)
    if params.after_reset_flip_probability > 0:
        append_anti_basis_error(head, data_qubits_x, params.after_reset_flip_probability, "X")
        append_anti_basis_error(head, data_qubits_z, params.after_reset_flip_probability, "Z")

    # Reset the check qubits, the checks are initialized in parallel with the datas
    head.append("RZ", measurement_qubits)
    if params.after_reset_flip_probability > 0:
        append_anti_basis_error(head, measurement_qubits, params.after_reset_flip_probability, basis="Z")

    # Apply Hadamard gates to checks
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

    # the two qubit gates for extracting the XZZX stabilizers are applied in 4 steps
    # 1. apply cnots
    head.append("TICK", [])
    # Add swap errors to the data and check qubits involved in the cnots, the rest will
    # idling in such step (get them close)
    if params.pswap_depolarization > 0:
        for jj in range(params.nswaps[0]):
            head.append("DEPOLARIZE1", list(set(cnot_targets[0]) & set(measurement_qubits)) , params.pswap_depolarization)
        for jj in range(params.nswaps[1]):
            head.append("DEPOLARIZE1", list(set(cnot_targets[0]) & set(data_qubits)) , params.pswap_depolarization)
            # Idling errors to unused checks while swaps occur (taylor for same as swap)
            if params.before_round_data_bias_probability[0] > 0:
                idling_fact = params.pswap_depolarization / params.after_clifford2_depolarization
                p = idling_fact * params.before_round_data_bias_probability[0]
                eta = params.before_round_data_bias_probability[1]
                p_x = p/(2*(1+eta))
                p_y = p/(2*(1+eta))
                p_z = p*eta / (1+eta)
                # We select the data_qubits and measurement_qubits that do not belong to the targets
                head.append("PAULI_CHANNEL_1", list((set(data_qubits) | set(measurement_qubits)) - (set(cnot_targets[0]))), [p_x, p_y, p_z])
    head.append("CNOT", cnot_targets[0])
    if params.after_clifford2_depolarization > 0:
        head.append("DEPOLARIZE2", cnot_targets[0], params.after_clifford2_depolarization)
    # Idling errors to unused checks (while CNOTs operating)
    # our baseline error is the two qubit gate not need to rescale idling here
    if params.before_round_data_bias_probability[0] > 0:
        p = params.before_round_data_bias_probability[0]
        eta = params.before_round_data_bias_probability[1]
        p_x = p/(2*(1+eta))
        p_y = p/(2*(1+eta))
        p_z = p*eta / (1+eta)
        # We select the data_qubits and measurement_qubits that do not belong to the targets
        head.append("PAULI_CHANNEL_1", list((set(data_qubits) | set(measurement_qubits)) - (set(cnot_targets[0]))), [p_x, p_y, p_z])
    # Add swap errors to the data and check qubits involved in the cnots, the rest will
    # idling in such step (go back to original position)
    if params.pswap_depolarization > 0:
        for jj in range(params.nswaps[0]):
            head.append("DEPOLARIZE1", list(set(cnot_targets[0]) & set(measurement_qubits)) , params.pswap_depolarization)
        for jj in range(params.nswaps[1]):
            head.append("DEPOLARIZE1", list(set(cnot_targets[0]) & set(data_qubits)) , params.pswap_depolarization)
            # Idling errors to unused checks while swaps occur (taylor for same as swap)
            if params.before_round_data_bias_probability[0] > 0:
                idling_fact = params.pswap_depolarization / params.after_clifford2_depolarization
                p = idling_fact * params.before_round_data_bias_probability[0]
                eta = params.before_round_data_bias_probability[1]
                p_x = p/(2*(1+eta))
                p_y = p/(2*(1+eta))
                p_z = p*eta / (1+eta)
                # We select the data_qubits and measurement_qubits that do not belong to the targets
                head.append("PAULI_CHANNEL_1", list((set(data_qubits) | set(measurement_qubits)) - (set(cnot_targets[0]))), [p_x, p_y, p_z])


    # 2. & 3. apply CZ gates
    for targets in cnot_targets[1:3]:
        head.append("TICK", [])
        # Add swap errors to the data and check qubits involved in the cnots, the rest will
        # idling in such step (get them close)
        if params.pswap_depolarization > 0:
            for jj in range(params.nswaps[0]):
                head.append("DEPOLARIZE1", list(set(targets) & set(measurement_qubits)) , params.pswap_depolarization)
            for jj in range(params.nswaps[1]):
                head.append("DEPOLARIZE1", list(set(targets) & set(data_qubits)) , params.pswap_depolarization)
                # Idling errors to unused checks while swaps occur (taylor for same as swap)
                if params.before_round_data_bias_probability[0] > 0:
                    idling_fact = params.pswap_depolarization / params.after_clifford2_depolarization
                    p = idling_fact * params.before_round_data_bias_probability[0]
                    eta = params.before_round_data_bias_probability[1]
                    p_x = p/(2*(1+eta))
                    p_y = p/(2*(1+eta))
                    p_z = p*eta / (1+eta)
                    # We select the data_qubits and measurement_qubits that do not belong to the targets
                    head.append("PAULI_CHANNEL_1", list((set(data_qubits) | set(measurement_qubits)) - (set(targets))), [p_x, p_y, p_z])
        head.append("CZ", targets)
        if params.after_clifford2_depolarization > 0:
            head.append("DEPOLARIZE2", targets, params.after_clifford2_depolarization)
        # Idling errors to unused checks which are biased (while CZs operating)
        # our baseline error is the two qubit gate we do not need to rescale the idling here
        if params.before_round_data_bias_probability[0] > 0:
            p = params.before_round_data_bias_probability[0]
            p_x = p/(2*(1+eta))
            p_y = p/(2*(1+eta))
            p_z = p*eta / (1+eta)
            # We select the data_qubits and measurement_qubits that do not belong to the targets
            head.append("PAULI_CHANNEL_1", list((set(data_qubits) | set(measurement_qubits)) - (set(targets))), [p_x, p_y, p_z])
        # Add swap errors to the data and check qubits involved in the cnots, the rest will
        # idling in such step (go back to original position)
        if params.pswap_depolarization > 0:
            for jj in range(params.nswaps[0]):
                head.append("DEPOLARIZE1", list(set(targets) & set(measurement_qubits)) , params.pswap_depolarization)
            for jj in range(params.nswaps[1]):
                head.append("DEPOLARIZE1", list(set(targets) & set(data_qubits)) , params.pswap_depolarization)
                # Idling errors to unused checks while swaps occur (taylor for same as swap)
                if params.before_round_data_bias_probability[0] > 0:
                    idling_fact = params.pswap_depolarization / params.after_clifford2_depolarization
                    p = idling_fact * params.before_round_data_bias_probability[0]
                    eta = params.before_round_data_bias_probability[1]
                    p_x = p/(2*(1+eta))
                    p_y = p/(2*(1+eta))
                    p_z = p*eta / (1+eta)
                    # We select the data_qubits and measurement_qubits that do not belong to the targets
                    head.append("PAULI_CHANNEL_1", list((set(data_qubits) | set(measurement_qubits)) - (set(targets))), [p_x, p_y, p_z])

    # 4. apply cnots
    head.append("TICK", [])
    # Add swap errors to the data and check qubits involved in the cnots, the rest will
    # idling in such step (get them close)
    if params.pswap_depolarization > 0:
        for jj in range(params.nswaps[0]):
            head.append("DEPOLARIZE1", list(set(cnot_targets[3]) & set(measurement_qubits)) , params.pswap_depolarization)
        for jj in range(params.nswaps[1]):
            head.append("DEPOLARIZE1", list(set(cnot_targets[3]) & set(data_qubits)) , params.pswap_depolarization)
            # Idling errors to unused checks while swaps occur (taylor for same as swap)
            if params.before_round_data_bias_probability[0] > 0:
                idling_fact = params.pswap_depolarization / params.after_clifford2_depolarization
                p = idling_fact * params.before_round_data_bias_probability[0]
                eta = params.before_round_data_bias_probability[1]
                p_x = p/(2*(1+eta))
                p_y = p/(2*(1+eta))
                p_z = p*eta / (1+eta)
                # We select the data_qubits and measurement_qubits that do not belong to the targets
                head.append("PAULI_CHANNEL_1", list((set(data_qubits) | set(measurement_qubits)) - (set(cnot_targets[3]))), [p_x, p_y, p_z])
    head.append("CNOT", cnot_targets[3])
    if params.after_clifford2_depolarization > 0:
        head.append("DEPOLARIZE2", cnot_targets[3], params.after_clifford2_depolarization)
    # Idling errors to unused checks (while CNOTs operating)
    #our baseline error is the two qubit gate we do not need to rescale idling here
    if params.before_round_data_bias_probability[0] > 0:
        p = params.before_round_data_bias_probability[0]
        eta = params.before_round_data_bias_probability[1]
        p_x = p/(2*(1+eta))
        p_y = p/(2*(1+eta))
        p_z = p*eta / (1+eta)
        # We select the data_qubits and measurement_qubits that do not belong to the targets
        head.append("PAULI_CHANNEL_1", list((set(data_qubits) | set(measurement_qubits)) - (set(cnot_targets[3]))), [p_x, p_y, p_z])
    # Add swap errors to the data and check qubits involved in the cnots, the rest will
    # idling in such step (go back to original position)
    if params.pswap_depolarization > 0:
        for jj in range(params.nswaps[0]):
            head.append("DEPOLARIZE1", list(set(cnot_targets[3]) & set(measurement_qubits)) , params.pswap_depolarization)
        for jj in range(params.nswaps[1]):
            head.append("DEPOLARIZE1", list(set(cnot_targets[3]) & set(data_qubits)) , params.pswap_depolarization)
            # Idling errors to unused checks while swaps occur (taylor for same as swap)
            if params.before_round_data_bias_probability[0] > 0:
                idling_fact = params.pswap_depolarization / params.after_clifford2_depolarization
                p = idling_fact * params.before_round_data_bias_probability[0]
                eta = params.before_round_data_bias_probability[1]
                p_x = p/(2*(1+eta))
                p_y = p/(2*(1+eta))
                p_z = p*eta / (1+eta)
                # We select the data_qubits and measurement_qubits that do not belong to the targets
                head.append("PAULI_CHANNEL_1", list((set(data_qubits) | set(measurement_qubits)) - (set(cnot_targets[3]))), [p_x, p_y, p_z])
    
    # Apply Hadamard gates to checks
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

    # Detectors
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
        if m_coord in chosen_basis_measure_coords:
            body.append(
                "DETECTOR",
                [stim.target_rec(-k - 1), stim.target_rec(-k - 1 - m)],
                [m_coord.real, m_coord.imag, 0.0]
            )
        
    #####--TAIL--####################################################
    # Build the end of the circuit, getting out of the cycle state and terminating.
    # In particular, the data measurements create detectors that have to be handled special.
    # Also, the tail is responsible for identifying the logical observable.
    tail = stim.Circuit()
    # Measure the data qubits. Similar to the reset, every other qubit is measured in the X basis and the rest in the Z basis.
    # Reset the check qubits
    tail.append("TICK", [])
    tail.append("RZ", measurement_qubits)
    if params.after_reset_flip_probability > 0:
        append_anti_basis_error(tail, measurement_qubits, params.after_reset_flip_probability, basis="Z")

    # Biased channel on the data qubits as per idling in this time step
    if params.before_round_data_bias_probability[0] > 0:
        idling_fact = params.after_reset_flip_probability / params.after_clifford2_depolarization
        p = idling_fact * params.before_round_data_bias_probability[0]
        eta = params.before_round_data_bias_probability[1]
        p_x = p/(2*(1+eta))
        p_y = p/(2*(1+eta))
        p_z = p*eta / (1+eta)
        tail.append("PAULI_CHANNEL_1", data_qubits, [p_x, p_y, p_z])
    
    # Apply Hadamard gates to checks
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

    # the two qubit gates for extracting the XZZX stabilizers are applied in 4 steps
    # 1. apply cnots
    tail.append("TICK", [])
    # Add swap errors to the data and check qubits involved in the cnots, the rest will
    # idling in such step (get them close)
    if params.pswap_depolarization > 0:
        for jj in range(params.nswaps[0]):
            tail.append("DEPOLARIZE1", list(set(cnot_targets[0]) & set(measurement_qubits)) , params.pswap_depolarization)
        for jj in range(params.nswaps[1]):
            tail.append("DEPOLARIZE1", list(set(cnot_targets[0]) & set(data_qubits)) , params.pswap_depolarization)
            # Idling errors to unused checks while swaps occur (taylor for same as swap)
            if params.before_round_data_bias_probability[0] > 0:
                idling_fact = params.pswap_depolarization / params.after_clifford2_depolarization
                p = idling_fact * params.before_round_data_bias_probability[0]
                eta = params.before_round_data_bias_probability[1]
                p_x = p/(2*(1+eta))
                p_y = p/(2*(1+eta))
                p_z = p*eta / (1+eta)
                # We select the data_qubits and measurement_qubits that do not belong to the targets
                tail.append("PAULI_CHANNEL_1", list((set(data_qubits) | set(measurement_qubits)) - (set(cnot_targets[0]))), [p_x, p_y, p_z])
    tail.append("CNOT", cnot_targets[0])
    if params.after_clifford2_depolarization > 0:
        tail.append("DEPOLARIZE2", cnot_targets[0], params.after_clifford2_depolarization)
    # Idling errors to unused checks (while CNOTs operating)
    # our baseline error rate is the two qubit gate we do not need to rescale idling here
    if params.before_round_data_bias_probability[0] > 0:
        p = params.before_round_data_bias_probability[0]
        eta = params.before_round_data_bias_probability[1]
        p_x = p/(2*(1+eta))
        p_y = p/(2*(1+eta))
        p_z = p*eta / (1+eta)
        # We select the data_qubits and measurement_qubits that do not belong to the targets
        tail.append("PAULI_CHANNEL_1", list((set(data_qubits) | set(measurement_qubits)) - (set(cnot_targets[0]))), [p_x, p_y, p_z])
    # Add swap errors to the data and check qubits involved in the cnots, the rest will
    # idling in such step (go back to original position)
    if params.pswap_depolarization > 0:
        for jj in range(params.nswaps[0]):
            tail.append("DEPOLARIZE1", list(set(cnot_targets[0]) & set(measurement_qubits)) , params.pswap_depolarization)
        for jj in range(params.nswaps[1]):
            tail.append("DEPOLARIZE1", list(set(cnot_targets[0]) & set(data_qubits)) , params.pswap_depolarization)
            # Idling errors to unused checks while swaps occur (taylor for same as swap)
            if params.before_round_data_bias_probability[0] > 0:
                idling_fact = params.pswap_depolarization / params.after_clifford2_depolarization
                p = idling_fact * params.before_round_data_bias_probability[0]
                eta = params.before_round_data_bias_probability[1]
                p_x = p/(2*(1+eta))
                p_y = p/(2*(1+eta))
                p_z = p*eta / (1+eta)
                # We select the data_qubits and measurement_qubits that do not belong to the targets
                tail.append("PAULI_CHANNEL_1", list((set(data_qubits) | set(measurement_qubits)) - (set(cnot_targets[0]))), [p_x, p_y, p_z])

    # 2. & 3. apply CZ gates with depolarizing noise
    for targets in cnot_targets[1:3]:
        tail.append("TICK", [])
        # Add swap errors to the data and check qubits involved in the cnots, the rest will
        # idling in such step (get them close)
        if params.pswap_depolarization > 0:
            for jj in range(params.nswaps[0]):
                tail.append("DEPOLARIZE1", list(set(targets) & set(measurement_qubits)) , params.pswap_depolarization)
            for jj in range(params.nswaps[1]):
                tail.append("DEPOLARIZE1", list(set(targets) & set(data_qubits)) , params.pswap_depolarization)
                # Idling errors to unused checks while swaps occur (taylor for same as swap)
                if params.before_round_data_bias_probability[0] > 0:
                    idling_fact = params.pswap_depolarization / params.after_clifford2_depolarization
                    p = idling_fact * params.before_round_data_bias_probability[0]
                    eta = params.before_round_data_bias_probability[1]
                    p_x = p/(2*(1+eta))
                    p_y = p/(2*(1+eta))
                    p_z = p*eta / (1+eta)
                    # We select the data_qubits and measurement_qubits that do not belong to the targets
                    tail.append("PAULI_CHANNEL_1", list((set(data_qubits) | set(measurement_qubits)) - (set(targets))), [p_x, p_y, p_z])
        tail.append("CZ", targets)
        if params.after_clifford2_depolarization > 0:
            tail.append("DEPOLARIZE2", targets, params.after_clifford2_depolarization)
        # Idling errors to unused checks which are biased (while CZs operating)
        # our baseline error is the two qubit gate we do not need to rescale idling here
        if params.before_round_data_bias_probability[0] > 0:
            p = params.before_round_data_bias_probability[0]
            p_x = p/(2*(1+eta))
            p_y = p/(2*(1+eta))
            p_z = p*eta / (1+eta)
            # We select the data_qubits and measurement_qubits that do not belong to the targets
            tail.append("PAULI_CHANNEL_1", list((set(data_qubits) | set(measurement_qubits)) - (set(targets))), [p_x, p_y, p_z])
        # Add swap errors to the data and check qubits involved in the cnots, the rest will
        # idling in such step (go back to original position)
        if params.pswap_depolarization > 0:
            for jj in range(params.nswaps[0]):
                tail.append("DEPOLARIZE1", list(set(targets) & set(measurement_qubits)) , params.pswap_depolarization)
            for jj in range(params.nswaps[1]):
                tail.append("DEPOLARIZE1", list(set(targets) & set(data_qubits)) , params.pswap_depolarization)
                # Idling errors to unused checks while swaps occur (taylor for same as swap)
                if params.before_round_data_bias_probability[0] > 0:
                    idling_fact = params.pswap_depolarization / params.after_clifford2_depolarization
                    p = idling_fact * params.before_round_data_bias_probability[0]
                    eta = params.before_round_data_bias_probability[1]
                    p_x = p/(2*(1+eta))
                    p_y = p/(2*(1+eta))
                    p_z = p*eta / (1+eta)
                    # We select the data_qubits and measurement_qubits that do not belong to the targets
                    tail.append("PAULI_CHANNEL_1", list((set(data_qubits) | set(measurement_qubits)) - (set(targets))), [p_x, p_y, p_z])
    

    # 4. apply cnots
    tail.append("TICK", [])
    # Add swap errors to the data and check qubits involved in the cnots, the rest will
    # idling in such step (get them close)
    if params.pswap_depolarization > 0:
        for jj in range(params.nswaps[0]):
            tail.append("DEPOLARIZE1", list(set(cnot_targets[3]) & set(measurement_qubits)) , params.pswap_depolarization)
        for jj in range(params.nswaps[1]):
            tail.append("DEPOLARIZE1", list(set(cnot_targets[3]) & set(data_qubits)) , params.pswap_depolarization)
            # Idling errors to unused checks while swaps occur (taylor for same as swap)
            if params.before_round_data_bias_probability[0] > 0:
                idling_fact = params.pswap_depolarization / params.after_clifford2_depolarization
                p = idling_fact * params.before_round_data_bias_probability[0]
                eta = params.before_round_data_bias_probability[1]
                p_x = p/(2*(1+eta))
                p_y = p/(2*(1+eta))
                p_z = p*eta / (1+eta)
                # We select the data_qubits and measurement_qubits that do not belong to the targets
                tail.append("PAULI_CHANNEL_1", list((set(data_qubits) | set(measurement_qubits)) - (set(cnot_targets[3]))), [p_x, p_y, p_z])
    tail.append("CNOT", cnot_targets[3])
    if params.after_clifford2_depolarization > 0:
        tail.append("DEPOLARIZE2", cnot_targets[3], params.after_clifford2_depolarization)
    # Idling errors to unused checks (while CNOTs operating)
    # our baseline error is the two qubit gate we do not need to rescale the idling here
    if params.before_round_data_bias_probability[0] > 0:
        p = params.before_round_data_bias_probability[0]
        eta = params.before_round_data_bias_probability[1]
        p_x = p/(2*(1+eta))
        p_y = p/(2*(1+eta))
        p_z = p*eta / (1+eta)
        # We select the data_qubits and measurement_qubits that do not belong to the targets
        tail.append("PAULI_CHANNEL_1", list((set(data_qubits) | set(measurement_qubits)) - (set(cnot_targets[3]))), [p_x, p_y, p_z])
    # Add swap errors to the data and check qubits involved in the cnots, the rest will
    # idling in such step (go back to original position)
    if params.pswap_depolarization > 0:
        for jj in range(params.nswaps[0]):
            tail.append("DEPOLARIZE1", list(set(cnot_targets[3]) & set(measurement_qubits)) , params.pswap_depolarization)
        for jj in range(params.nswaps[1]):
            tail.append("DEPOLARIZE1", list(set(cnot_targets[3]) & set(data_qubits)) , params.pswap_depolarization)
            # Idling errors to unused checks while swaps occur (taylor for same as swap)
            if params.before_round_data_bias_probability[0] > 0:
                idling_fact = params.pswap_depolarization / params.after_clifford2_depolarization
                p = idling_fact * params.before_round_data_bias_probability[0]
                eta = params.before_round_data_bias_probability[1]
                p_x = p/(2*(1+eta))
                p_y = p/(2*(1+eta))
                p_z = p*eta / (1+eta)
                # We select the data_qubits and measurement_qubits that do not belong to the targets
                tail.append("PAULI_CHANNEL_1", list((set(data_qubits) | set(measurement_qubits)) - (set(cnot_targets[3]))), [p_x, p_y, p_z])
    
    # Apply Hadamard gates to checks
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

    # Measure the check qubits in the Z basis
    tail.append("TICK", [])    
    if params.before_measure_flip_probability > 0:
        append_anti_basis_error(tail, measurement_qubits, params.before_measure_flip_probability, basis="Z")
    tail.append("M" + "Z", measurement_qubits)

    # Detectors of checks
    m = len(measurement_qubits)
    tail.append("SHIFT_COORDS", [], [0.0, 0.0, 1.0])
    for m_index in measurement_qubits:
        m_coord = q2p[m_index]
        k = len(measurement_qubits) - measure_coord_to_order[m_coord] - 1
        if m_coord in chosen_basis_measure_coords:
            tail.append(
                "DETECTOR",
                [stim.target_rec(-k - 1), stim.target_rec(-k - 1 - m)],
                [m_coord.real, m_coord.imag, 0.0]
            )
    
    # Measure the data qubits. Similar to the reset, every other qubit is measured in the X basis and the rest in the Z basis.
    if params.before_measure_flip_probability > 0:
        append_anti_basis_error(tail, data_qubits_x, params.before_measure_flip_probability, "X")
        append_anti_basis_error(tail, data_qubits_z, params.before_measure_flip_probability, "Z")  
    for q in data_qubits:
        # this keeps the order of measuring the data qubits consistent with the order of the data qubits
        tail.append("M" + "XZ"[q in data_qubits_z], [q])
        
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
       
    return head + body * (params.rounds - 2) + tail
        