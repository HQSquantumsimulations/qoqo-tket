# Copyright © 2024 HQS Quantum Simulations GmbH. All Rights Reserved.
# License details given in distributed LICENSE file.

"""package to compile and run qoqo programms with tket."""

from qoqo import Circuit, QuantumProgram
from qoqo_qasm import QasmBackend, qasm_str_to_circuit
from pytket.qasm import circuit_from_qasm_str, circuit_to_qasm_str
from pytket.backends import Backend
from pytket.backends.backendresult import BackendResult
from typing import List, Union
from qoqo.measurements import (  # type:ignore
    PauliZProduct,
    ClassicalRegister,
    CheatedPauliZProduct,
    Cheated,
)


def compile_with_tket(
    circuits: Union[Circuit, List[Circuit]], backend: Backend
) -> Circuit:
    """Use a tket backend to compile qoqo circuit(s).

    Args:
        circuits (Union[Circuit, List[Circuit]]): qoqo circuit(s)
        backend (Backend): tket backend

    Returns:
        Circuit: compiled qoqo circuit
    """
    circuits_is_list = isinstance(circuits, list)
    circuits = circuits if circuits_is_list else [circuits]

    qasm_backend = QasmBackend(qasm_version="2.0")

    tket_circuits = [
        circuit_from_qasm_str(qasm_backend.circuit_to_qasm_str(circuit))
        for circuit in circuits
    ]
    compiled_tket_circuits = backend.get_compiled_circuits(tket_circuits)

    tket_qasm = [
        circuit_to_qasm_str(compiled_tket_circuit).replace("( ", " ")
        for compiled_tket_circuit in compiled_tket_circuits
    ]

    transpiled_qoqo_circuits = [
        qasm_str_to_circuit(qasm_str) for qasm_str in tket_qasm
    ]
    return (
        transpiled_qoqo_circuits
        if circuits_is_list
        else transpiled_qoqo_circuits[0]
    )


def run_with_tket(
    circuits: Union[Circuit, list[Circuit]],
    backend: Backend,
    n_shots: Union[int, list[int], None] = None,
) -> Union[BackendResult, List[BackendResult]]:
    """Use a tket backend to run qoqo circuit(s).

    Args:
        circuits (Union[Circuit, list[Circuit]]): qoqo circuit(s)
        backend (Backend): tket backend
        n_shots (Union[int, list[int], None]): number of shots for each circuit

    Returns:
        Union[BackendResult, List[BackendResult]]: Result for each circuit
    """
    circuits_is_list = isinstance(circuits, list)
    circuits = circuits if circuits_is_list else [circuits]

    qasm_backend = QasmBackend(qasm_version="2.0")

    tket_circuits = [
        circuit_from_qasm_str(qasm_backend.circuit_to_qasm_str(circuit))
        for circuit in circuits
    ]
    compiled_tket_circuits = backend.get_compiled_circuits(tket_circuits)
    for circuit in tket_circuits:
        circuit.measure_all()

    tket_results = backend.run_circuits(compiled_tket_circuits, n_shots)

    return tket_results if circuits_is_list else tket_results[0]


def compile_program_with_tket(
    quantum_program: QuantumProgram, backend: Backend
) -> QuantumProgram:
    """Use tket backend to compile a QuantumProgram.

    Args:
        quantum_program (QuantumProgram): QuantumProgram to transpile.
        backend (Backend): backend to use.

    Returns:
        QuantumProgram: transpiled QuantumProgram.
    """
    constant_circuit = quantum_program.measurement().constant_circuit()
    circuits = quantum_program.measurement().circuits()
    circuits = (
        circuits
        if constant_circuit is None
        else [constant_circuit + circuit for circuit in circuits]
    )
    transpiled_circuits = compile_with_tket(circuits, backend)

    def recreate_measurement(
        quantum_program: QuantumProgram, transpiled_circuits: List[Circuit]
    ) -> Union[
        PauliZProduct, ClassicalRegister, CheatedPauliZProduct, Cheated
    ]:
        """Recreate a measurement QuantumProgram using the transpiled circuits.

        Args:
            quantum_program (QuantumProgram): quantumProgram to transpile.
            transpiled_circuits (List[Circuit]): transpiled circuits.

        Returns:
            Union[PauliZProduct, ClassicalRegister,
            CheatedPauliZProduct, Cheated]: measurement

        Raises:
            TypeError: if the measurement type is not supported.
        """
        if isinstance(quantum_program.measurement(), PauliZProduct):
            return PauliZProduct(
                constant_circuit=None,
                circuits=transpiled_circuits,
                input=quantum_program.measurement().input(),
            )
        elif isinstance(quantum_program.measurement(), CheatedPauliZProduct):
            return CheatedPauliZProduct(
                constant_circuit=None,
                circuits=transpiled_circuits,
                input=quantum_program.measurement().input(),
            )
        elif isinstance(quantum_program.measurement(), Cheated):
            return Cheated(
                constant_circuit=None,
                circuits=transpiled_circuits,
                input=quantum_program.measurement().input(),
            )
        elif isinstance(quantum_program.measurement(), ClassicalRegister):
            return ClassicalRegister(
                constant_circuit=None, circuits=transpiled_circuits
            )
        else:
            raise TypeError("Unknown measurement type")

    return QuantumProgram(
        measurement=recreate_measurement(quantum_program, transpiled_circuits),
        input_parameter_names=quantum_program.input_parameter_names(),
    )
