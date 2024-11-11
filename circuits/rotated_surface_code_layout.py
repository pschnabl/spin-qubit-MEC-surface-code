from typing import Set, List, Dict
import math

def generate_rotated_surface_code_circuit_layout(distance: int = None,
                                                 x_distance: int = None,
                                                 z_distance: int = None):
    """
    Generate the layout of a rotated surface code with the given distance.

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
            - cnot_targets (List[List[int]]): List of lists of qubit indices representing the CNOT gate targets.
            - measure_coord_to_order (Dict[complex, int]): Dictionary mapping complex numbers to integers indicating the order of the measurement qubits.
            - data_coord_to_order (Dict[complex, int]): Dictionary mapping complex numbers to integers indicating the order of the data qubits.
            - z_order (List[complex]): List of complex numbers representing the z-order of the qubits.
    """
    if distance is not None and x_distance is None and z_distance is None:
        # check if distance is at least 3
        if distance < 3:
            raise ValueError("Distance must be at least 3.")
        # x and z distance treated similarly for now.
        x_distance = distance
        z_distance = distance
    
    elif x_distance is not None and z_distance is not None and distance is None:
        if x_distance < 3 or z_distance < 3:
            raise ValueError("x_distance and z_distance must be at least 3.")
 
    else:
        raise ValueError("Exactly one of distance or (x_distance and z_distance) must be provided.")
        

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

    # Define interaction orders so that hook errors run against the error grain instead of with it.
    z_order: List[complex] = [1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]
    x_order: List[complex] = [1 + 1j, -1 + 1j, 1 - 1j, -1 - 1j]

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

    # List out CNOT gate targets using given interaction orders.
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
                cnot_targets[k].append(p2q[data])
                cnot_targets[k].append(p2q[measure])
                
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