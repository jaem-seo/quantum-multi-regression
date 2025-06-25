import numpy as np
import torch
import pennylane as qml
import matplotlib.pyplot as plt
import pickle

torch.manual_seed(0)
np.random.seed(0)
torch.set_default_dtype(torch.float64)

n_epochs = 20
lr = 0.1
init_level = 0.5
noise = 0.1
eps = 1e-5
batch = 64
n_ensemble = 10

# ---------------- data ----------------
N = 200
x = torch.linspace(-np.pi, np.pi, N, requires_grad=False)
y1 = torch.sin(x) + noise * torch.randn_like(x)
y2 = torch.cos(x) + noise * torch.randn_like(x)
y3 = -torch.cos(x) + noise * torch.randn_like(x)

# --------------- Traditional QNN ------------------
n_qubits, depth = 2, 3
dev_exp = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev_exp, interface="torch")
def circuit_exp(params, x_val):
    # input encoding
    qml.AngleEmbedding([x_val], wires=[0, 1])
    # rotation and entanglement
    qml.StronglyEntanglingLayers(params, wires=[0, 1])
    return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1)) # maximum 2-vector

class QNN_Exp(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.theta = torch.nn.Parameter(init_level * torch.randn(depth, n_qubits, 3))

    def forward(self, x_batch):
        outs = [torch.stack(circuit_exp(self.theta, xi)) for xi in x_batch]
        return torch.stack(outs)  # shape (B, 2)

# --------------- Log-Ratio Probability QNN --------
dev_prob = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev_prob, interface="torch")
def circuit_prob(params, x_val):
    qml.AngleEmbedding([x_val], wires=[0, 1])
    qml.StronglyEntanglingLayers(params, wires=[0, 1])
    return qml.probs(wires=[0, 1])  # 4-vector

class QNN_LRP(torch.nn.Module):
    def __init__(self, eps=eps):
        super().__init__()
        self.theta = torch.nn.Parameter(init_level * torch.randn(depth, n_qubits, 3))
        self.eps = eps

    def forward(self, x_batch):
        outs = []
        for xi in x_batch:
            probs = torch.clamp(circuit_prob(self.theta, xi), self.eps, 1-self.eps)      # p0,p1,p2,p3
            ratios = torch.log(probs[:-1] / probs[-1])  # 3 outputs
            outs.append(ratios)
        return torch.stack(outs)  # shape (B, 3)

# ---------------- training routine ----------------
def train(model, targets, n_epochs=50, lr=0.1):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_hist = []
    for ep in range(n_epochs):
        perm = torch.randperm(N)
        epoch_loss = 0.
        for i in range(0, N, batch):
            idx = perm[i:i+batch]
            opt.zero_grad()
            out = model(x[idx])[:, :targets.shape[-1]]
            loss = torch.mean((out - targets[idx]) ** 2)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(idx)
        loss_hist.append(epoch_loss / N)
        print(f'Ep {ep}: loss {epoch_loss / N}')
    return loss_hist, model

# ----------- Train Traditional (2 outputs) --------
losses_exp, models_exp = [], []
targets_exp = torch.stack([y1, y2], dim=1)  # (N,2)
for i in range(n_ensemble):
    loss_exp_hist, model_exp = train(QNN_Exp(), targets_exp, n_epochs=n_epochs, lr=lr)
    losses_exp.append(loss_exp_hist)
    models_exp.append(model_exp)

# ----------- Train LRP (2 outputs) ----------------
losses_lrp2, models_lrp2 = [], []
targets_lrp2 = torch.stack([y1, y2], dim=1)  # (N,2)
for i in range(n_ensemble):
    loss_lrp2_hist, model_lrp2 = train(QNN_LRP(), targets_lrp2, n_epochs=n_epochs, lr=lr)
    losses_lrp2.append(loss_lrp2_hist)
    models_lrp2.append(model_lrp2)
    
# ----------- Train LRP (3 outputs) ----------------
losses_lrp3, models_lrp3 = [], []
targets_lrp3 = torch.stack([y1, y2, y3], dim=1)  # (N,3)
for i in range(n_ensemble):
    loss_lrp3_hist, model_lrp3 = train(QNN_LRP(), targets_lrp3, n_epochs=n_epochs, lr=lr)
    losses_lrp3.append(loss_lrp3_hist)
    models_lrp3.append(model_lrp3)

# Save results
data = {
    'losses_exp': losses_exp,
    'models_exp': models_exp,
    'losses_lrp2': losses_lrp2,
    'models_lrp2': models_lrp2,
    'losses_lrp3': losses_lrp3,
    'models_lrp3': models_lrp3,
}
with open('data.pickle', 'wb') as f:
    pickle.dump(data, f)

# Load results
with open('data.pickle', 'rb') as f:
    data = pickle.load(f)

losses_exp = data['losses_exp']
models_exp = data['models_exp']
losses_lrp2 = data['losses_lrp2']
models_lrp2 = data['models_lrp2']
losses_lrp3 = data['losses_lrp3']
models_lrp3 = data['models_lrp3']


# ------------------- plots ------------------------
def fill_plot(
    ys,
    xs=None,
    ax=None,
    line_kw=None,
    fill_kw=None,
):
    # to array
    ys = np.asarray(ys, dtype=float)
    mean = ys.mean(axis=0)
    std  = ys.std(axis=0)

    if xs is None:
        xs = np.arange(1, mean.size + 1)

    # styles
    if line_kw is None:
        line_kw = dict(lw=2)            
    if fill_kw is None:
        fill_kw = dict(alpha=0.25)      

    if ax is None:
        fig, ax = plt.subplots()

    ax.fill_between(xs, mean - std, mean + std, **fill_kw)
    ax.plot(xs, mean, **line_kw)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    return ax

fig, ax = plt.subplots(1, 1, figsize=(3, 2))
ax = fill_plot(losses_exp, ax=ax, line_kw=dict(lw=2, c='chocolate'), fill_kw=dict(alpha=0.25, color='chocolate'))
ax = fill_plot(losses_lrp2, ax=ax, line_kw=dict(lw=2, c='g'), fill_kw=dict(alpha=0.25, color='g'))
ax = fill_plot(losses_lrp3, ax=ax, line_kw=dict(lw=2, c='royalblue'), fill_kw=dict(alpha=0.25, color='royalblue'))
ax.set_yscale('log')
plt.tight_layout()
plt.savefig('losses.svg')


fig, axes = plt.subplots(3, 1, sharex=True, figsize=(3, 3))

# predictions
x_np = x.detach().numpy()
with torch.no_grad():
    preds_exp = np.array([m(x).numpy() for m in models_exp])
    preds_lrp2 = np.array([m(x).numpy() for m in models_lrp2])
    preds_lrp3 = np.array([m(x).numpy() for m in models_lrp3])

axes[0].scatter(x_np, y1, s=3, color="k", label="true")
axes[1].scatter(x_np, y2, s=3, color="k", label="true")
axes[2].scatter(x_np, y3, s=3, color="k", label="true")

for i in range(2):
    axes[i] = fill_plot(preds_exp[:, :, i], x_np, ax=axes[i], line_kw=dict(lw=2, c='chocolate'), fill_kw=dict(alpha=0.25, color='chocolate'))
    axes[i] = fill_plot(preds_lrp2[:, :, i], x_np, ax=axes[i], line_kw=dict(lw=2, c='g'), fill_kw=dict(alpha=0.25, color='g'))
    axes[i] = fill_plot(preds_lrp3[:, :, i], x_np, ax=axes[i], line_kw=dict(lw=2, c='royalblue'), fill_kw=dict(alpha=0.25, color='royalblue'))

axes[2] = fill_plot(preds_lrp3[:, :, 2], x_np, ax=axes[2], line_kw=dict(lw=2, c='royalblue'), fill_kw=dict(alpha=0.25, color='royalblue'))

plt.tight_layout()
plt.savefig('fits.svg')

plt.show()

