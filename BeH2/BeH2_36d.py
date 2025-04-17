from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister,transpile
from qiskit import Aer, execute
from qiskit.visualization import plot_histogram
from qiskit.tools.visualization import plot_histogram, plot_state_city
from qiskit.tools.monitor import job_monitor
from azure.quantum.qiskit import AzureQuantumProvider
from qiskit.providers.aer.noise import NoiseModel, depolarizing_error, pauli_error
from qiskit.utils import QuantumInstance

import numpy as np
from scipy.linalg import block_diag
from scipy.optimize import minimize
from scipy.optimize import OptimizeResult, approx_fprime
import math
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--trial',type= int, default=0)
parser.add_argument('--shots',type= int)
parser.add_argument('--eps',type= float, default=0.02)
parser.add_argument('--init',type= float, default=1.5)
args = parser.parse_args()


def func_and_gradient(x_opt, fun, eps):
    f = fun(x_opt)
    grad = np.zeros_like(x_opt)
    print(eps)
    for i in range(x_opt.size):
        x_plus_h = x_opt.copy()
        x_plus_h[i] += eps
        f_plus_h = fun(x_plus_h)
        grad[i] = (f_plus_h - f) / eps
    return f, grad


def adam(fun, x0, jac=None, iters=100, options=None, beta1=0.9, beta2=0.999, epsilon=1e-8, eps=0.02):
    
    # learning rate
    base_lr = 0.1

    # Initialize the ADAM variables
    t = 0
    m_t = 0
    v_t = 0

    # Initialize the optimization result
    nit = 0
    nfev = 0
    njev = 0
    success = False
    message = ''
    fun_val = np.inf
    x_opt = np.copy(x0)

    if options is None:
        options = {}
    maxiter = options.get('maxiter', iters)

    while nit < maxiter:
        # Compute the function value and gradient
        fval, grad = func_and_gradient(x_opt, fun, eps)  # approx_fprime(x_opt, fun, eps)
        print(nit, x_opt)
        nfev += 2
        njev += 1

        fun_val = fval
        # Compute the ADAM update
        t += 1
        m_t = beta1 * m_t + (1 - beta1) * grad
        v_t = beta2 * v_t + (1 - beta2) * grad ** 2
        m_t_hat = m_t / (1 - beta1 ** t)
        v_t_hat = v_t / (1 - beta2 ** t)

        lr_current = 0.5 * base_lr * (math.cos(math.pi * nit / maxiter) + 1)
        lr = lr_current / (np.sqrt(v_t_hat) + epsilon)

        # Update the parameters
        x_opt = x_opt - lr * m_t_hat

        nit += 1

    result = OptimizeResult(fun=fun_val, x=x_opt, nit=nit, nfev=nfev, njev=njev, success=success, message=message)

    return result


class history:
    def __init__(self):
        self.thetas = []
        self.energies = []
        self.shots = []
        
def vqe_simulator(theta, shots, history):
    NUM_SHOTS = shots
    Energy = 0.0

    def get_probability_distribution_6(counts):
        # Ensure all possible 6-qubit outcomes are in counts
        for i in range(2**6):  # 2**6 = 64 for a 6-qubit system
            k = format(i, '06b')  # Format i as a 6-bit binary string
            if k not in counts:
                counts[k] = 0
        
        # Sort counts and calculate probabilities
        sorted_counts = sorted(counts.items())
        output_distr = [v[1] / NUM_SHOTS for v in sorted_counts]
        
        return output_distr

    def calculate_pauli_expectation(pauli_string, output_distr):
        expectation = 0
        for i, prob in enumerate(output_distr):
            outcome = format(i, '06b')  # Convert index to binary string representing measurement outcome
            contribution = 1
            for op, bit in zip(pauli_string, outcome):
                if op == 'Z':
                    contribution *= 1 if bit == '0' else -1
                # Note: 'I' does not change the expectation value
            expectation += contribution * prob
        return expectation

    def ansatz_circuit(theta, group):
        qc = QuantumCircuit(6)
        # Initial state HF: |111000>
        qc.x(0)
        qc.x(1)
        qc.x(2)
        # Apply rotations and entanglements (2 layers)
        for i in range(6):
        	qc.ry(theta[i], i)

        for i in range(6, 12):
        	qc.rz(theta[i], i % 6)

        for i in range(5):
        	qc.cx(i, i + 1)

        # Anoter layer
        for i in range(12, 18):
        	qc.ry(theta[i], i % 6)

        for i in range(18, 24):
        	qc.rz(theta[i], i % 6)

        for i in range(5):
        	qc.cx(i, i + 1)


        # Apply basis transformation based on the Pauli group
        for i in range(6):
            if any(pauli_string[i] == 'X' for pauli_string in group):
                qc.h(i)
            elif any(pauli_string[i] == 'Y' for pauli_string in group):
                qc.sdg(i)
                qc.h(i)
        return qc

    # Pauli strings and coefficients for a 6-qubit system
    strings_1 = ['IIIIII', 'ZIIIII', 'ZZIIII', 'IIZIII', 'IZZIII', 'IIZZII', 'IIIIZI', 'IIIZZI', 'IZIIII', 'ZIZIII', 'ZZZIII', 'ZIZZII', 'ZIIIZI', 'ZIIZZI', 'ZZZZII', 'ZZIIZI', 'ZZIZZI', 'IIZIII', 'IIZZII', 'IIZIZI', 'IIZZZI', 'IZZIII', 'IZZZII', 'IZZIZI', 'IZZZZI', 'IIIZII']
    coeff_1 = [-12.603278277226423, 2.024163057794482, 0.1500572309132946, -0.015112493494256966, -0.1277837028755862, -0.031011347099158516, -0.015112493494257001, 0.031011347099158627, -0.1040500634744448, -0.057914286959194254, 0.08933316120497292, 0.138667863329013, 0.14236052837839772, -0.138667863329013, 0.07712697264945595, 0.08933316120497292, -0.07712697264945595, 0.08197746851667174, 0.08523660616595644, -0.09427772585578935, 0.10034007066108214, -0.08523660616595644, -0.08197746851667174, -0.08523660616595644, 0.08197746851667174, -0.11246476027166769]
	# Construct and execute the quantum circuit for each group of Pauli strings
    qc = ansatz_circuit(theta, strings_1)
    qc.measure_all()
    job = execute(qc, Aer.get_backend('qasm_simulator'), shots=NUM_SHOTS)
    result = job.result()
    counts = result.get_counts(qc)
    output_distr = get_probability_distribution_6(counts)

    # Calculate energy contribution from this group of Pauli strings
    E1 = sum(coeff_1[i] * calculate_pauli_expectation(strings_1[i], output_distr) for i in range(len(strings_1)))
    
    strings_2 = ['XZIIII', 'XIIIII', 'XZZIII', 'XIZIII', 'XZZZII', 'XIZZII', 'XZIIZI', 'XIIIZI', 'XZIZZI', 'XIIZZI']
    coeff_2 = [-0.07464968129196993, 0.07464968129196993, 5.379378185769383e-05, -5.379378185769383e-05, 0.003523356858926417, -0.003523356858926417, 5.379378185769383e-05, -5.379378185769383e-05, -0.003523356858926417, 0.003523356858926417]

    qc = ansatz_circuit(theta, strings_2)
    qc.measure_all()
    job = execute(qc, Aer.get_backend('qasm_simulator'), shots=NUM_SHOTS)
    result = job.result()
    counts = result.get_counts(qc)
    output_distr = get_probability_distribution_6(counts)

    # Calculate energy contribution from this group of Pauli strings
    E2 = sum(coeff_2[i] * calculate_pauli_expectation(strings_2[i], output_distr) for i in range(len(strings_2)))

    strings_3 = ['IIYZII', 'IZYIII']
    coeff_3 = [0.003259137649284688, -0.003259137649284688]

    qc = ansatz_circuit(theta, strings_3)
    qc.measure_all()
    job = execute(qc, Aer.get_backend('qasm_simulator'), shots=NUM_SHOTS)
    result = job.result()
    counts = result.get_counts(qc)
    output_distr = get_probability_distribution_6(counts)

    # Calculate energy contribution from this group of Pauli strings
    E3 = sum(coeff_3[i] * calculate_pauli_expectation(strings_3[i], output_distr) for i in range(len(strings_3)))

    strings_4 = ['IZXIII', 'IIXZII']
    coeff_4 = [-0.003259137649284688, 0.003259137649284688]

    qc = ansatz_circuit(theta, strings_4)
    qc.measure_all()
    job = execute(qc, Aer.get_backend('qasm_simulator'), shots=NUM_SHOTS)
    result = job.result()
    counts = result.get_counts(qc)
    output_distr = get_probability_distribution_6(counts)

    # Calculate energy contribution from this group of Pauli strings
    E4 = sum(coeff_4[i] * calculate_pauli_expectation(strings_4[i], output_distr) for i in range(len(strings_4)))

    strings_5 = ['IIYZYI', 'IZYIYI']
    coeff_5 = [0.0060623448052927915, -0.003259137649284688]

    qc = ansatz_circuit(theta, strings_5)
    qc.measure_all()
    job = execute(qc, Aer.get_backend('qasm_simulator'), shots=NUM_SHOTS)
    result = job.result()
    counts = result.get_counts(qc)
    output_distr = get_probability_distribution_6(counts)

    # Calculate energy contribution from this group of Pauli strings
    E5 = sum(coeff_5[i] * calculate_pauli_expectation(strings_5[i], output_distr) for i in range(len(strings_5)))

    strings_6 = ['IZXIXI', 'IIXZXI']
    coeff_6 = [-0.003259137649284688, -0.0060623448052927915]

    qc = ansatz_circuit(theta, strings_6)
    qc.measure_all()
    job = execute(qc, Aer.get_backend('qasm_simulator'), shots=NUM_SHOTS)
    result = job.result()
    counts = result.get_counts(qc)
    output_distr = get_probability_distribution_6(counts)

    # Calculate energy contribution from this group of Pauli strings
    E6 = sum(coeff_6[i] * calculate_pauli_expectation(strings_6[i], output_distr) for i in range(len(strings_6)))

    Energy = E1 + E2 + E3 + E4 + E5 + E6

    history.energies.append(Energy)
    history.shots.append(NUM_SHOTS*6)
    
    print(history.energies[-1], sum(history.shots))
    return Energy


def vqe_simulator(theta, shots, history):
    NUM_SHOTS = shots
    Energy = 0.0

    def get_probability_distribution_6(counts):
        # Ensure all possible 6-qubit outcomes are in counts
        for i in range(2**6):  # 2**6 = 64 for a 6-qubit system
            k = format(i, '06b')  # Format i as a 6-bit binary string
            if k not in counts:
                counts[k] = 0
        
        # Sort counts and calculate probabilities
        sorted_counts = sorted(counts.items())
        output_distr = [v[1] / NUM_SHOTS for v in sorted_counts]
        
        return output_distr

    def calculate_pauli_expectation(pauli_string, output_distr):
        expectation = 0
        for i, prob in enumerate(output_distr):
            outcome = format(i, '06b')  # Convert index to binary string representing measurement outcome
            contribution = 1
            for op, bit in zip(pauli_string, outcome):
                if op == 'Z':
                    contribution *= 1 if bit == '0' else -1
                # Note: 'I' does not change the expectation value
            expectation += contribution * prob
        return expectation

    def ansatz_circuit(theta, group):
        qc = QuantumCircuit(6)
        # Initial state HF: |111000>
        qc.x(0)
        qc.x(1)
        qc.x(2)
        # Apply rotations and entanglements (3 layers)
        for i in range(6):
        	qc.ry(theta[i], i)

        for i in range(5):
        	qc.cx(i, i + 1)

        # Anoter layer
        for i in range(6, 12):
        	qc.ry(theta[i], i % 6)

        for i in range(5):
        	qc.cx(i, i + 1)

        # The third layer
        for i in range(12, 18):
            qc.ry(theta[i], i % 6)

        for i in range(5):
            qc.cx(i, i + 1)

        # The fourth layer
        for i in range(18, 24):
            qc.ry(theta[i], i % 6)

        for i in range(5):
            qc.cx(i, i + 1)

        # The fifth layer
        for i in range(24, 30):
            qc.ry(theta[i], i % 6)

        for i in range(5):
            qc.cx(i, i + 1)

        # The sixth layer
        for i in range(30, 36):
            qc.ry(theta[i], i % 6)

        for i in range(5):
            qc.cx(i, i + 1)


        # Apply basis transformation based on the Pauli group
        for i in range(6):
            if any(pauli_string[i] == 'X' for pauli_string in group):
                qc.h(i)
            elif any(pauli_string[i] == 'Y' for pauli_string in group):
                qc.sdg(i)
                qc.h(i)
        return qc

    # Pauli strings and coefficients for a 6-qubit system
    strings_1 = ['IIIIII', 'ZIIIII', 'ZZIIII', 'IIZIII', 'IZZIII', 'IIZZII', 'IIIIZI', 'IIIZZI', 'IZIIII', 'ZIZIII', 'ZZZIII', 'ZIZZII', 'ZIIIZI', 'ZIIZZI', 'ZZZZII', 'ZZIIZI', 'ZZIZZI', 'IIZIII', 'IIZZII', 'IIZIZI', 'IIZZZI', 'IZZIII', 'IZZZII', 'IZZIZI', 'IZZZZI', 'IIIZII']
    coeff_1 = [-12.603278277226423, 2.024163057794482, 0.1500572309132946, -0.015112493494256966, -0.1277837028755862, -0.031011347099158516, -0.015112493494257001, 0.031011347099158627, -0.1040500634744448, -0.057914286959194254, 0.08933316120497292, 0.138667863329013, 0.14236052837839772, -0.138667863329013, 0.07712697264945595, 0.08933316120497292, -0.07712697264945595, 0.08197746851667174, 0.08523660616595644, -0.09427772585578935, 0.10034007066108214, -0.08523660616595644, -0.08197746851667174, -0.08523660616595644, 0.08197746851667174, -0.11246476027166769]
	# Construct and execute the quantum circuit for each group of Pauli strings
    qc = ansatz_circuit(theta, strings_1)
    qc.measure_all()
    job = execute(qc, Aer.get_backend('qasm_simulator'), shots=NUM_SHOTS)
    result = job.result()
    counts = result.get_counts(qc)
    output_distr = get_probability_distribution_6(counts)

    # Calculate energy contribution from this group of Pauli strings
    E1 = sum(coeff_1[i] * calculate_pauli_expectation(strings_1[i], output_distr) for i in range(len(strings_1)))
    
    strings_2 = ['XZIIII', 'XIIIII', 'XZZIII', 'XIZIII', 'XZZZII', 'XIZZII', 'XZIIZI', 'XIIIZI', 'XZIZZI', 'XIIZZI']
    coeff_2 = [-0.07464968129196993, 0.07464968129196993, 5.379378185769383e-05, -5.379378185769383e-05, 0.003523356858926417, -0.003523356858926417, 5.379378185769383e-05, -5.379378185769383e-05, -0.003523356858926417, 0.003523356858926417]

    qc = ansatz_circuit(theta, strings_2)
    qc.measure_all()
    job = execute(qc, Aer.get_backend('qasm_simulator'), shots=NUM_SHOTS)
    result = job.result()
    counts = result.get_counts(qc)
    output_distr = get_probability_distribution_6(counts)

    # Calculate energy contribution from this group of Pauli strings
    E2 = sum(coeff_2[i] * calculate_pauli_expectation(strings_2[i], output_distr) for i in range(len(strings_2)))

    strings_3 = ['IIYZII', 'IZYIII']
    coeff_3 = [0.003259137649284688, -0.003259137649284688]

    qc = ansatz_circuit(theta, strings_3)
    qc.measure_all()
    job = execute(qc, Aer.get_backend('qasm_simulator'), shots=NUM_SHOTS)
    result = job.result()
    counts = result.get_counts(qc)
    output_distr = get_probability_distribution_6(counts)

    # Calculate energy contribution from this group of Pauli strings
    E3 = sum(coeff_3[i] * calculate_pauli_expectation(strings_3[i], output_distr) for i in range(len(strings_3)))

    strings_4 = ['IZXIII', 'IIXZII']
    coeff_4 = [-0.003259137649284688, 0.003259137649284688]

    qc = ansatz_circuit(theta, strings_4)
    qc.measure_all()
    job = execute(qc, Aer.get_backend('qasm_simulator'), shots=NUM_SHOTS)
    result = job.result()
    counts = result.get_counts(qc)
    output_distr = get_probability_distribution_6(counts)

    # Calculate energy contribution from this group of Pauli strings
    E4 = sum(coeff_4[i] * calculate_pauli_expectation(strings_4[i], output_distr) for i in range(len(strings_4)))

    strings_5 = ['IIYZYI', 'IZYIYI']
    coeff_5 = [0.0060623448052927915, -0.003259137649284688]

    qc = ansatz_circuit(theta, strings_5)
    qc.measure_all()
    job = execute(qc, Aer.get_backend('qasm_simulator'), shots=NUM_SHOTS)
    result = job.result()
    counts = result.get_counts(qc)
    output_distr = get_probability_distribution_6(counts)

    # Calculate energy contribution from this group of Pauli strings
    E5 = sum(coeff_5[i] * calculate_pauli_expectation(strings_5[i], output_distr) for i in range(len(strings_5)))

    strings_6 = ['IZXIXI', 'IIXZXI']
    coeff_6 = [-0.003259137649284688, -0.0060623448052927915]

    qc = ansatz_circuit(theta, strings_6)
    qc.measure_all()
    job = execute(qc, Aer.get_backend('qasm_simulator'), shots=NUM_SHOTS)
    result = job.result()
    counts = result.get_counts(qc)
    output_distr = get_probability_distribution_6(counts)

    # Calculate energy contribution from this group of Pauli strings
    E6 = sum(coeff_6[i] * calculate_pauli_expectation(strings_6[i], output_distr) for i in range(len(strings_6)))

    Energy = E1 + E2 + E3 + E4 + E5 + E6
    
    history.energies.append(Energy)
    history.shots.append(NUM_SHOTS*6)
    print(history.energies[-1], sum(history.shots))
    
    return Energy

    
if __name__ == '__main__':
    theta_test = [args.init]*36

    record_history = history()
    per_shots = args.shots//6

    # ADAM optimizer
    result = adam(lambda x: vqe_simulator(x, shots=per_shots, history=record_history), theta_test, iters=500, eps=args.eps)
    print(result)

    # np.savez('test.npz', theta=thetas, energy=energies)
    np.savez('results36d/qiskit_adam24d_shots{}_init_{}-eps{}-iter500-coslr-trial{}.npz'.format(args.shots, args.init, args.eps, args.trial),
        theta=record_history.thetas, energy=record_history.energies)