import torch, numpy as np, matplotlib.pyplot as plt, pennylane as qml
torch.set_default_dtype(torch.float64)

# Settings
n_qubits, depth = 4, 5          # qubits × depth
grid_pts, span = 61, np.pi      # contour resolution
lr_iters = 60                   # power-iteration
device = torch.device("cpu")

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Tools
def flatten(params):
    return torch.cat([p.view(-1) for p in params])

def unflatten(vec, like_params):
    shapes, out, idx = [p.shape for p in like_params], [], 0
    for s in shapes:
        numel = np.prod(s)
        out.append(vec[idx:idx+numel].view(*s))
        idx += numel
    return out

def hvp(loss, params, vec):
    """Hessian-vector product via double backward."""
    grad = torch.autograd.grad(loss, params, create_graph=True)
    grad_vec = torch.dot(flatten(grad), vec)
    hv = torch.autograd.grad(grad_vec, params, retain_graph=True)
    return flatten(hv)

def top_k_eigen(model, loss_fn, k=2, iters=60):
    """Power iteration + deflation to get top-k eigenpairs."""
    params = list(model.parameters())
    v_list, λ_list = [], []
    for _ in range(k):
        v = torch.randn_like(flatten(params))
        v /= v.norm()
        for _ in range(iters):
            model.zero_grad(set_to_none=True)
            loss = loss_fn()
            hv = hvp(loss, params, v)
            for v_prev in v_list:
                hv -= hv.dot(v_prev) * v_prev
            v = hv / (hv.norm() + 1e-12)
        model.zero_grad(set_to_none=True)
        λ = (hvp(loss_fn(), params, v)).dot(v)
        v_list.append(v.detach())
        λ_list.append(λ.detach())
    return λ_list, v_list

# Quantum circuits and models
def build_qnode_exp(nq, d):
    dev = qml.device("default.qubit", wires=nq, shots=None)
    @qml.qnode(dev, interface='torch')
    def circuit(weights):
        weights = weights.reshape(d, nq)
        for layer in range(d):
            for w in range(nq):
                qml.RY(weights[layer, w], wires=w)
            for w in range(nq-1):
                qml.CNOT(wires=[w, w+1])
        return qml.expval(qml.PauliZ(0))
    return circuit

def build_qnode_log(nq, d):
    dev = qml.device("default.qubit", wires=nq, shots=None)
    @qml.qnode(dev, interface='torch')
    def circuit(weights):
        weights = weights.reshape(d, nq)
        for layer in range(d):
            for w in range(nq):
                qml.RY(weights[layer, w], wires=w)
            for w in range(nq-1):
                qml.CNOT(wires=[w, w+1])
        return qml.probs(wires=range(nq))
    return circuit

class ExpModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.theta = torch.nn.Parameter(torch.rand(depth, n_qubits)*2*np.pi)
        self.qnode = build_qnode_exp(n_qubits, depth)
    def forward(self): return self.qnode(self.theta)

class LogModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.theta = torch.nn.Parameter(torch.rand(depth, n_qubits)*2*np.pi)
        self.qnode = build_qnode_log(n_qubits, depth)
    def forward(self, eps=1e-5):
        p = torch.clamp(self.qnode(self.theta), eps, 1-eps)
        r = torch.log(p[:-1] / p[-1])        # 2^n-1 log-ratio
        return r[0].mean()                   # A scalar output for example

models  = [ExpModel().to(device), LogModel().to(device)]
targets = [torch.tensor(0.), torch.tensor(0.)]       # Arbitrary target
loss_fns = [
    lambda m=models[0], t=targets[0]: (m() - t) ** 2,
    lambda m=models[1], t=targets[1]: (m() - t) ** 2,
]
titles = ["Pauli-Z Expectation", "Log-ratio Probability"]

# 3D Plots
fig, axes = plt.subplots(1, 2, figsize=(7, 3), subplot_kw={"projection": "3d"})

for ax, model, loss_fn, title in zip(axes, models, loss_fns, titles):
    λs, vs = top_k_eigen(model, loss_fn, k=2, iters=lr_iters)
    v1, v2 = vs[0], vs[1]
    v1 /= v1.norm(); v2 -= v2.dot(v1)*v1; v2 /= v2.norm()  # 정규/직교

    alphas = torch.linspace(-span, span, grid_pts)
    betas  = torch.linspace(-span, span, grid_pts)
    loss_map = np.zeros((grid_pts, grid_pts))

    θ0 = flatten(model.parameters()).detach()

    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            δ = a * v1 + b * v2
            new_vec = θ0 + δ
            with torch.no_grad():
                for p, upd in zip(model.parameters(), unflatten(new_vec, model.parameters())):
                    p.copy_(upd)
            loss_map[i, j] = loss_fn().item()

    with torch.no_grad():
        for p, orig in zip(model.parameters(), unflatten(θ0, model.parameters())):
            p.copy_(orig)

    A, B = np.meshgrid(alphas, betas)
    cs = ax.plot_surface(A, B, np.log(loss_map), cmap="viridis")
    ax.set_xlabel(r"$\theta_1$"); ax.set_ylabel(r"$\theta_2$")

    zticks = ['1e-15', '1e-10', '1e-5', '1e0', '1e5']
    zticks_num = [1e-15, 1e-10, 1e-5, 1e0, 1e5]
    ax.set_zticks(np.log10(zticks_num))
    ax.set_zticklabels(zticks)

plt.tight_layout()
plt.savefig('landscape.svg')
plt.show()

