import pennylane as qml
from pennylane import numpy as np
import torch, matplotlib.pyplot as plt
import pickle
from copy import deepcopy

torch.set_default_dtype(torch.float64)
np.random.seed(0); torch.manual_seed(0)

eps = 1e-5 
init_level = 0.1
n_ensemble, n_epochs, batch, lr = 10, 200, 64, 0.1

# ------------------------------------------------------------
# 1.  Synthetic noisy data: y = sin(x) + ε
# ------------------------------------------------------------
dx = np.pi / 100
x_data = torch.Tensor(np.append(np.arange(0., 0.5*np.pi, dx), np.arange(np.pi, 2*np.pi, dx)))
noise_level = 0.25 * torch.clamp(-torch.sin(x_data), 0, None) ** 2
y_true = torch.sin(x_data)
y_noise = y_true + noise_level * torch.randn_like(y_true)
N = len(x_data)

# ------------------------------------------------------------
# 2.  LRP-QNN (2-qubit, AngleEmbedding + SELayers)
# ------------------------------------------------------------
n_qubits, n_layers = 2, 3
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def circuit_prob(weights, x, eps=eps):
    qml.AngleEmbedding([x], wires=[0, 1])
    qml.StronglyEntanglingLayers(weights, wires=[0, 1])
    return qml.probs(wires=[0, 1])      # just return the probabilities

class LRPQNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = torch.nn.Parameter(init_level * torch.randn(n_layers, n_qubits, 3))

    def forward(self, x_batch):
        mu, sigma = [], []
        for xi in x_batch:
            probs = circuit_prob(self.weights, xi)
            probs = torch.clamp(probs, eps, 1-eps)
            ratios = torch.log(probs[:-1] / probs[-1])
            mu.append(ratios[0])
            sigma.append(torch.exp(ratios[1]))
        return torch.stack(mu), torch.stack(sigma)

# Negative log-likelihood loss
def nll_loss(mu, sigma, target):
    sigma = torch.clamp(sigma, eps, 1/eps)
    return torch.mean(0.5 * np.log(2 * np.pi) + torch.log(sigma) +
                      0.5 * ((target - mu) / sigma)**2)

# ------------------------------------------------------------
# 3.  Train an ensemble of 10 models
# ------------------------------------------------------------
models, histories = [], []

for seed in range(n_ensemble):
    torch.manual_seed(seed)
    model = LRPQNN()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist = []
    best_model, best_loss = None, np.inf
    for ep in range(n_epochs):
        perm = torch.randperm(N)
        epoch_loss = 0.
        for i in range(0, N, batch):
            idx = perm[i:i+batch]
            opt.zero_grad()
            mu_hat, sig_hat = model(x_data)
            loss = nll_loss(mu_hat, sig_hat, y_noise)
            loss.backward(); opt.step()
            epoch_loss += loss.item() * len(idx) / N
        hist.append(epoch_loss)
        print(f'Seed {seed} / Ep {ep}: loss {epoch_loss}')
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model = deepcopy(model)

    models.append(best_model); histories.append(hist)

    data = {
        'models': models,
        'histories': histories
    }
    with open('data.pickle', 'wb') as f:
        pickle.dump(data, f)

with open('data.pickle', 'rb') as f:
    data = pickle.load(f)

models, histories = data['models'], data['histories']

# ------------------------------------------------------------
# 4.   Inference:  μ̄ , σ̄, uncertainties
# ------------------------------------------------------------
with torch.no_grad():
    mus   = torch.stack([m(x_data)[0] for m in models])          # (E,N)
    sigs2 = torch.stack([(m(x_data)[1]**2) for m in models])     # (E,N)

epistemic   = mus.var(dim=0).sqrt()              # σ_epi = √Var[μ]
aleatoric   = sigs2.mean(dim=0).sqrt()           # σ_ale = √E[σ²]
total_unc   = (epistemic**2 + aleatoric**2).sqrt()

pred_mu     = mus.mean(dim=0)

# ------------------------------------------------------------
# 5.  Plots
# ------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13,4))

# (a) Loss histories
for hist in histories:
    axes[0].plot(hist, alpha=0.4, color="C1")
axes[0].set_title("Ensemble training curves (NLL)")
axes[0].set_xlabel("epoch"); axes[0].set_ylabel("NLL"); axes[0].set_yscale("symlog")

# (b) Prediction & uncertainty bands
x_np = x_data.numpy()
axes[1].plot(x_np, y_noise.numpy(), '.', ms=3, alpha=0.4, label="noisy data")
axes[1].plot(x_np, y_true.numpy(), 'k--', lw=1, label="true sin(x)")
axes[1].plot(x_np, pred_mu.numpy(), 'C0', lw=2, label="pred. mean μ")

# Uncertainty bands
axes[1].fill_between(x_np, (pred_mu-aleatoric).numpy(), (pred_mu+aleatoric).numpy(),
                     color="C2", alpha=0.25, label="aleatoric ±1σ")
axes[1].fill_between(x_np, (pred_mu-total_unc).numpy(), (pred_mu+total_unc).numpy(),
                     color="C1", alpha=0.15, label="total ±1σ")

axes[1].set_title("Prediction with epistemic & aleatoric uncertainty")
axes[1].set_xlabel("x"); axes[1].legend(loc="upper right", fontsize=8)

plt.tight_layout(); plt.show()

