# Copyright © 2024 HQS Quantum Simulations GmbH. All Rights Reserved.
"""Test everything."""
from qoqo_tket.qoqo_tket import compile_with_tket, run_with_tket, compile_program_with_tket
import numpy as np
from qoqo import Circuit, CircuitDag, QuantumProgram
from qoqo.measurements import (  # type:ignore
    PauliZProduct,
    PauliZProductInput,
    ClassicalRegister,
    CheatedPauliZProduct,
    CheatedPauliZProductInput,
    Cheated,
    CheatedInput,
)
from qoqo import operations as ops
from pytket.extensions.projectq import ProjectQBackend
from pytket.circuit import BasisOrder


def test_compile_qoqo_tket() -> None:
    """Test compiling with qoqo_tket."""
    circuit = Circuit()
    circuit += ops.Identity(0)
    circuit += ops.PauliX(0)

    circuit_res = Circuit()
    circuit_res += ops.RotateX(0, 3.141592653589793)

    backend = ProjectQBackend()
    compiled_circuit = compile_with_tket(circuit, backend)
    assert compiled_circuit == circuit_res


def test_compile_complex_qoqo_tket() -> None:
    """Test compiling with qoqo_tket."""
    circuit = Circuit()
    circuit += ops.Hadamard(0)
    circuit += ops.CNOT(0, 1)
    circuit += ops.PauliX(1)
    circuit += ops.CNOT(1, 2)
    circuit += ops.PauliZ(2)

    circuit_res = Circuit()
    circuit_res += ops.RotateZ(0, 10.995574287564276)
    circuit_res += ops.RotateX(0, 4.71238898038469)
    circuit_res += ops.RotateZ(0, 1.5707963267948966)
    circuit_res += ops.RotateZ(1, 9.42477796076938)
    circuit_res += ops.RotateX(1, 3.141592653589793)
    circuit_res += ops.RotateZ(2, 3.141592653589793)
    circuit_res += ops.CNOT(0, 1)
    circuit_res += ops.CNOT(1, 2)

    backend = ProjectQBackend()
    compiled_circuit = compile_with_tket(circuit, backend)
    compiled_circuit_dag = CircuitDag()
    circuit_res_dag = CircuitDag()
    compiled_circuit_dag = compiled_circuit_dag.from_circuit(compiled_circuit)
    circuit_res_dag = circuit_res_dag.from_circuit(circuit_res)
    assert compiled_circuit_dag == circuit_res_dag


def test_compile_multiple_qoqo_tket() -> None:
    """Test compiling with qoqo_tket."""
    circuit = Circuit()
    circuit += ops.Identity(0)
    circuit += ops.PauliX(0)

    circuit_res = Circuit()
    circuit_res += ops.RotateX(0, 3.141592653589793)

    circuit_2 = Circuit()
    circuit_2 += ops.Hadamard(0)
    circuit_2 += ops.CNOT(0, 1)
    circuit_2 += ops.PauliX(1)
    circuit_2 += ops.CNOT(1, 2)
    circuit_2 += ops.PauliZ(2)

    circuit_res_2 = Circuit()
    circuit_res_2 += ops.RotateZ(0, 10.995574287564276)
    circuit_res_2 += ops.RotateX(0, 4.71238898038469)
    circuit_res_2 += ops.RotateZ(0, 1.5707963267948966)
    circuit_res_2 += ops.RotateZ(1, 9.42477796076938)
    circuit_res_2 += ops.RotateX(1, 3.141592653589793)
    circuit_res_2 += ops.RotateZ(2, 3.141592653589793)
    circuit_res_2 += ops.CNOT(0, 1)
    circuit_res_2 += ops.CNOT(1, 2)

    backend = ProjectQBackend()
    compiled_circuits = compile_with_tket([circuit, circuit_2], backend)

    compiled_circuit_dag = CircuitDag()
    circuit_res_dag = CircuitDag()
    compiled_circuit_dag = compiled_circuit_dag.from_circuit(compiled_circuits[1])
    circuit_res_dag = circuit_res_dag.from_circuit(circuit_res_2)

    assert compiled_circuits[0] == circuit_res and compiled_circuit_dag == circuit_res_dag


def test_run_qoqo_tket() -> None:
    """Test compiling with qoqo_tket."""
    circuit = Circuit()
    circuit += ops.PauliX(0)
    circuit += ops.DefinitionBit("ro", 1, True)
    circuit += ops.MeasureQubit(0, "ro", 0)

    state_res = [0, 1]

    backend = ProjectQBackend()
    results = run_with_tket(circuit, backend)

    assert np.isclose(results.get_state(), state_res, atol=1e-5).all()


def test_run_complex_qoqo_tket() -> None:
    """Test compiling with qoqo_tket."""
    circuit = Circuit()
    circuit += ops.Hadamard(0)
    circuit += ops.CNOT(0, 1)
    circuit += ops.PauliX(1)
    circuit += ops.CNOT(1, 2)
    circuit += ops.PauliZ(2)

    state_res = [0, 1 / np.sqrt(2), 0, 0, 0, 0, -1 / np.sqrt(2), 0]

    backend = ProjectQBackend()
    results = run_with_tket(circuit, backend)
    assert np.isclose(results.get_state(basis=BasisOrder.dlo), state_res, atol=1e-5).all()


def test_run_multiple_qoqo_tket() -> None:
    """Test compiling with qoqo_tket."""
    circuit = Circuit()
    circuit += ops.PauliX(0)
    circuit += ops.DefinitionBit("ro", 1, True)
    circuit += ops.MeasureQubit(0, "ro", 0)

    state_res = [0, 1]

    circuit_2 = Circuit()
    circuit_2 += ops.Hadamard(0)
    circuit_2 += ops.CNOT(0, 1)
    circuit_2 += ops.PauliX(1)
    circuit_2 += ops.CNOT(1, 2)
    circuit_2 += ops.PauliZ(2)

    state_res_2 = [0, 1 / np.sqrt(2), 0, 0, 0, 0, -1 / np.sqrt(2), 0]

    backend = ProjectQBackend()
    results = run_with_tket([circuit, circuit_2], backend)

    assert (
        np.isclose(results[0].get_state(basis=BasisOrder.dlo), state_res, atol=1e-5).all()
        and np.isclose(results[1].get_state(basis=BasisOrder.dlo), state_res_2, atol=1e-5).all()
    )


def assert_quantum_program_equal(
    quantum_program_1: QuantumProgram, quantum_program2: QuantumProgram
) -> None:
    """Assert that two quantum programs are equal.

    Args:
        quantum_program_1 (QuantumProgram): quantum program
        quantum_program2 (QuantumProgram): quantum program

    Raises:
        AssertionError: if the quantum programs are not equal
    """
    assert quantum_program_1.input_parameter_names() == quantum_program2.input_parameter_names()
    if not isinstance(quantum_program_1.measurement(), ClassicalRegister):
        assert quantum_program_1.measurement().input() == quantum_program2.measurement().input()
    assert (
        quantum_program_1.measurement().constant_circuit()
        == quantum_program2.measurement().constant_circuit()
    )
    for circuit_1, circuit_2 in zip(
        quantum_program_1.measurement().circuits(), quantum_program2.measurement().circuits()
    ):
        circuit_dag_1 = CircuitDag()
        circuit_dag_2 = CircuitDag()
        circuit_dag_1 = circuit_dag_1.from_circuit(circuit_1)
        circuit_dag_2 = circuit_dag_2.from_circuit(circuit_2)
        assert circuit_dag_1 == circuit_dag_2


def test_quantum_program() -> None:
    """Test basic program conversion with a BaseGates transpiler."""
    circuit_1 = Circuit()
    circuit_1 += ops.PauliX(0)
    circuit_1 += ops.Identity(0)

    circuit_res_1 = Circuit()
    circuit_res_1 += ops.RotateX(0, 3.141592653589793)

    measurement_input = CheatedPauliZProductInput()
    measurement = CheatedPauliZProduct(
        constant_circuit=None, circuits=[circuit_1], input=measurement_input
    )
    measurement_res = CheatedPauliZProduct(
        constant_circuit=None,
        circuits=[circuit_res_1],
        input=measurement_input,
    )
    quantum_program = QuantumProgram(measurement=measurement, input_parameter_names=["x"])
    quantum_program_res = QuantumProgram(measurement=measurement_res, input_parameter_names=["x"])

    backend = ProjectQBackend()
    transpiled_program = compile_program_with_tket(quantum_program, backend)

    assert_quantum_program_equal(transpiled_program, quantum_program_res)


def test_quantum_program_cheated() -> None:
    """Test basic program conversion with a BaseGates transpiler."""
    circuit_1 = Circuit()
    circuit_1 += ops.PauliX(0)
    circuit_1 += ops.Identity(0)

    circuit_res_1 = Circuit()
    circuit_res_1 += ops.RotateX(0, 3.141592653589793)

    measurement_input = CheatedInput(1)
    measurement = Cheated(constant_circuit=None, circuits=[circuit_1], input=measurement_input)
    measurement_res = Cheated(
        constant_circuit=None,
        circuits=[circuit_res_1],
        input=measurement_input,
    )
    quantum_program = QuantumProgram(measurement=measurement, input_parameter_names=["x"])
    quantum_program_res = QuantumProgram(measurement=measurement_res, input_parameter_names=["x"])

    backend = ProjectQBackend()
    transpiled_program = compile_program_with_tket(quantum_program, backend)

    assert_quantum_program_equal(transpiled_program, quantum_program_res)


def test_quantum_program_no_constant_circuit() -> None:
    """Test basic program conversion with a BaseGates transpiler."""
    circuit_1 = Circuit()
    circuit_1 += ops.PauliX(0)
    circuit_1 += ops.Identity(0)

    circuit_res_1 = Circuit()
    circuit_res_1 += ops.RotateX(0, 3.141592653589793)

    circuit_2 = Circuit()
    circuit_2 += ops.Hadamard(0)
    circuit_2 += ops.CNOT(0, 1)
    circuit_2 += ops.PauliX(1)
    circuit_2 += ops.CNOT(1, 2)
    circuit_2 += ops.PauliZ(2)

    circuit_res_2 = Circuit()
    circuit_res_2 += ops.RotateZ(0, 10.995574287564276)
    circuit_res_2 += ops.RotateX(0, 4.71238898038469)
    circuit_res_2 += ops.RotateZ(0, 1.5707963267948966)
    circuit_res_2 += ops.RotateZ(1, 9.42477796076938)
    circuit_res_2 += ops.RotateX(1, 3.141592653589793)
    circuit_res_2 += ops.RotateZ(2, 3.141592653589793)
    circuit_res_2 += ops.CNOT(0, 1)
    circuit_res_2 += ops.CNOT(1, 2)

    measurement_input = PauliZProductInput(1, False)
    measurement = PauliZProduct(
        constant_circuit=None, circuits=[circuit_1, circuit_2], input=measurement_input
    )
    measurement_res = PauliZProduct(
        constant_circuit=None,
        circuits=[circuit_res_1, circuit_res_2],
        input=measurement_input,
    )
    quantum_program = QuantumProgram(measurement=measurement, input_parameter_names=["x"])
    quantum_program_res = QuantumProgram(measurement=measurement_res, input_parameter_names=["x"])

    backend = ProjectQBackend()
    transpiled_program = compile_program_with_tket(quantum_program, backend)

    assert_quantum_program_equal(transpiled_program, quantum_program_res)


def test_quantum_programwith_constant_circuit() -> None:
    """Test basic program conversion with a BaseGates transpiler."""
    constant_circuit = Circuit()
    constant_circuit += ops.Hadamard(0)
    constant_circuit += ops.Hadamard(1)

    circuit_1 = Circuit()
    circuit_1 += ops.PauliX(0)
    circuit_1 += ops.Identity(0)

    circuit_res_1 = Circuit()
    circuit_res_1 += ops.RotateZ(0, 10.995574287564276)
    circuit_res_1 += ops.RotateX(0, 1.5707963267948966)
    circuit_res_1 += ops.RotateZ(0, 1.5707963267948966)
    circuit_res_1 += ops.RotateZ(1, 1.5707963267948966)
    circuit_res_1 += ops.RotateX(1, 1.5707963267948966)
    circuit_res_1 += ops.RotateZ(1, 1.5707963267948966)

    circuit_2 = Circuit()
    circuit_2 += ops.Hadamard(0)
    circuit_2 += ops.CNOT(0, 1)
    circuit_2 += ops.PauliX(1)
    circuit_2 += ops.CNOT(1, 2)
    circuit_2 += ops.PauliZ(2)

    circuit_res_2 = Circuit()
    circuit_res_2 += ops.RotateZ(0, 3.141592653589793)
    circuit_res_2 += ops.RotateZ(1, 1.5707963267948966)
    circuit_res_2 += ops.RotateX(1, 4.71238898038469)
    circuit_res_2 += ops.RotateZ(1, 1.5707963267948966)
    circuit_res_2 += ops.RotateZ(2, 3.141592653589793)
    circuit_res_2 += ops.CNOT(0, 1)
    circuit_res_2 += ops.CNOT(1, 2)

    measurement = ClassicalRegister(
        constant_circuit=constant_circuit, circuits=[circuit_1, circuit_2]
    )
    measurement_res = ClassicalRegister(
        constant_circuit=None,
        circuits=[circuit_res_1, circuit_res_2],
    )
    quantum_program = QuantumProgram(measurement=measurement, input_parameter_names=["x"])
    quantum_program_res = QuantumProgram(measurement=measurement_res, input_parameter_names=["x"])

    backend = ProjectQBackend()
    transpiled_program = compile_program_with_tket(quantum_program, backend)

    assert_quantum_program_equal(transpiled_program, quantum_program_res)
