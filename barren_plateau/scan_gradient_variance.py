import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

seed = 42
np.random.seed(seed)

ntry = 5
samples = 100

# Simple quantum circuit
def build_circuit(n_qubits, depth):
    def layer(params):
        # params.shape == (depth, n_qubits)
        for d in range(depth):
            for i in range(n_qubits):
                qml.RY(params[d, i], wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
    return layer

# 1) Pauli-Z expectation approach
def gradient_variance_expectation(n_qubits, depth, samples=100):
    dev = qml.device("default.qubit", wires=n_qubits)
    layer = build_circuit(n_qubits, depth)

    @qml.qnode(dev)
    def qnn(params):
        layer(params)
        return qml.expval(qml.PauliZ(0))       

    def expectation_loss(params):
        exps = qnn(params)
        return exps ** 2

    vars_ = []
    for _ in range(samples):
        params = np.random.uniform(0, 2*np.pi, (depth, n_qubits), requires_grad=True)
        grad = qml.grad(expectation_loss)(params)              # shape = (depth, n_qubits)
        vars_.append(np.mean(grad**2))           
    return np.array(vars_)

# 2) Log-ratio probability approach
def gradient_variance_logratio(n_qubits, depth, samples=100):
    dev = qml.device("default.qubit", wires=n_qubits)
    layer = build_circuit(n_qubits, depth)

    @qml.qnode(dev)
    def qnn_probs(params):
        layer(params)
        return qml.probs(wires=range(n_qubits)) 

    def log_ratio_loss(params, eps=1e-5):
        probs = np.clip(qnn_probs(params), eps, 1-eps)
        ratios = np.log(probs[:-1] / probs[-1])
        return np.mean(ratios**2)

    grad_fn = qml.grad(log_ratio_loss)

    vars_ = []
    for _ in range(samples):
        params = np.random.uniform(0, 2*np.pi, (depth, n_qubits), requires_grad=True)
        grad = grad_fn(params)
        vars_.append(np.mean(grad**2))
    return np.array(vars_)


y11, y12, y21, y22 = [], [], [], []

# Scan circuit depth
depth_list = [1, 2, 3, 4, 5]
nq_fixed = 4

for _ in range(ntry):
    var_exp_depth = np.array([gradient_variance_expectation(nq_fixed, d, samples=samples)
                     for d in depth_list])
    var_log_depth = np.array([gradient_variance_logratio(nq_fixed, d, samples=samples)
                     for d in depth_list])

    y12.append(var_exp_depth.mean(axis=1))
    y22.append(var_log_depth.mean(axis=1))


# Plots
fig, ax = plt.subplots(1, 1, sharey=True, figsize=(3, 1.5))

for i in range(ntry):
    ax.plot(depth_list, y12[i], c='chocolate', marker='o', markersize=3, lw=0.5, alpha=0.3)

ax.plot(depth_list, np.mean(y12, axis=0), c='chocolate', marker='o')

ax.set_yscale('log')
ax.set_ylim([1e-2, 2e-1])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.tight_layout()
plt.savefig('trad_plat.svg')


fig, ax = plt.subplots(1, 1, sharey=True, figsize=(3, 1.5))

for i in range(ntry):
    ax.plot(depth_list, y22[i], c='g', marker='o', markersize=3, lw=0.5, alpha=0.3)

ax.plot(depth_list, np.mean(y22, axis=0), c='g', marker='o')

ax.set_yscale('log')
ax.set_ylim([1e3, 5e4])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
    
plt.tight_layout()
plt.savefig('lrp_plat.svg')

plt.show()
