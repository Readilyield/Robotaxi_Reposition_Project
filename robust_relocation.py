import os, warnings
# 1) force C++ backend and suppress CVXPY warnings
os.environ["CVXPY_BACKEND"] = "cpp"
warnings.filterwarnings(
    "ignore",
    message=r"Constraint #\d+ contains too many subexpressions"
)
warnings.filterwarnings(
    "ignore",
    message=r"The problem has an expression with dimension greater than 2"
)

import torch
import torch.nn.functional as F
import cvxpy as cp
import numpy as np
from scipy import sparse
from cvxpylayers.torch import CvxpyLayer

#---------------------------
# 0) Device and Data Load
#---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



with np.load('trip_counts.npz') as data:
    trip_counts = data['trip_counts']
    num_dates = data['num_dates']

with np.load('mu_cp.npz') as data:
    mu = data['mu']

# mask trip_counts by 1 where 0
trip_counts[trip_counts == 0] = 1

# compute arrival rate
lambda_ = trip_counts.sum(axis=2) / (20 / 60 * num_dates) / 8000

# normalize trip_counts
P = trip_counts / trip_counts.sum(axis=2, keepdims=True)

lambda_all = torch.asarray(lambda_, dtype=torch.float32).unsqueeze_(0).to(device)
P_all = torch.asarray(P, dtype=torch.float32).unsqueeze_(0).to(device)
mu_all = torch.asarray(mu, dtype=torch.float32).unsqueeze_(0).to(device)



# hyperparameters
r: int = 16  # number of regions
T: int = 1 # number of time-slots a day
n_omegas: int = 5  # number of scenarios (e.g., 365 days).
window_size : int = 4  # look-ahead window size


lambda_all = (
    lambda_all[:, : n_omegas * T, :r]      # (1 , 6 , r)
    .contiguous()                         # make memory contiguous after the slice
    .view(n_omegas, T, r)                 # (n_omegas , T , r)
)

P_all  = (
    P_all[:, : n_omegas * T, :r, :r]        # (1 , 6 , r , r)
    .contiguous()
    .view(n_omegas, T, r, r)
)
P_all  = P_all / P_all.sum(dim=3, keepdim=True)

mu_all = (
    mu_all[:, : n_omegas * T, :r, :r]       # (1 , 6 , r , r)
    .contiguous()
    .view(n_omegas, T, r, r)
)

raw_Q = torch.nn.Parameter(torch.randn(T, r, r, device=device), requires_grad=True)

def row_softmax(M):
    return torch.softmax(M, dim=2)

#---------------------------
# 1) GPU moving average
#---------------------------
def compute_lookahead_all(lambda_all, mu_all, P_all, W):
    S, T, r = lambda_all.shape
    N = S * r
    # sliding window over time
    kernel1 = torch.ones(N, 1, W, device=lambda_all.device) / W
    lam = lambda_all.permute(0, 2, 1).reshape(1, N, T)
    lam_conv = F.conv1d(F.pad(lam, (0, W-1)), kernel1, groups=N)
    lambda_avg_all = lam_conv.squeeze(0).reshape(S, r, T).permute(0, 2, 1)

    def avg4(x4):
        N2 = S * r * r
        kernel2 = torch.ones(N2, 1, W, device=x4.device) / W
        x = x4.permute(0, 2, 3, 1).reshape(1, N2, T)
        x_conv = F.conv1d(F.pad(x, (0, W-1)), kernel2, groups=N2)
        return x_conv.squeeze(0).reshape(S, r, r, T).permute(0, 3, 1, 2)

    mu_avg_all = avg4(mu_all)
    P_avg_all  = avg4(P_all)
    return lambda_avg_all, mu_avg_all, P_avg_all

lambda_avg_all, mu_avg_all, P_avg_all = compute_lookahead_all(lambda_all, mu_all, P_all, window_size)

#---------------------------------------
# 2) CVXPYLayer builder
#---------------------------------------
def build_robust_routing_cvxpy(lambda_avg_all, mu_avg_all, P_avg_all):
    S, T, r = lambda_avg_all.shape
    N_A  = T * r
    N_EF = T * r * r

    # Variables & parameter
    Qp     = cp.Parameter((T, r, r))
    eta    = cp.Variable()
    A_flat = cp.Variable((S, N_A),  nonneg=True)
    E_flat = cp.Variable((S, N_EF), nonneg=True)
    F_flat = cp.Variable((S, N_EF), nonneg=True)

    def idx(t, i, j):
        return t * (r * r) + i * r + j

    total_vars = S*(N_A + 2*N_EF)

    # (a) Ride‐flow
    ride_rows, ride_cols, ride_data = [], [], []
    row_ctr = 0
    f_off = S*N_A + S*N_EF
    for s in range(S):
        for t in range(T):
            for i in range(r):
                for j in range(r):
                    # λ·P·A
                    ride_rows.append(row_ctr)
                    ride_cols.append(s*N_A + t*r + i)
                    ride_data.append(lambda_avg_all[s, t, i]*P_avg_all[s, t, i, j])
                    # -μ·F
                    ride_rows.append(row_ctr)
                    ride_cols.append(f_off + s*N_EF + idx(t, i, j))
                    ride_data.append(-mu_avg_all[s, t, i, j])
                    row_ctr += 1
    RideMat = sparse.coo_matrix((ride_data, (ride_rows, ride_cols)), shape=(row_ctr, total_vars))

    # (b) Empty‐flow
    e_off = S*N_A
    f_off = S*N_A + S*N_EF
    emptyE_rows = []
    emptyE_cols = []
    emptyE_data = []
    emptyF_rows = []
    emptyF_cols = []
    emptyF_data = []
    Q_empty_list = []
    row_ctr_e = 0
    for s in range(S):
        for t in range(T):
            for i in range(r):
                for j in range(r):
                    if i==j: continue
                    emptyE_rows.append(row_ctr_e)
                    emptyE_cols.append(e_off + s*N_EF + idx(t, i, j))
                    emptyE_data.append(mu_avg_all[s, t, i, j])
                    emptyF_rows.append(row_ctr_e)
                    emptyF_cols.append(f_off + s*N_EF + idx(t, i, j))
                    emptyF_data.append(mu_avg_all[s, t, i, j])
                    Q_empty_list.append(Qp[t, i, j])
                    row_ctr_e += 1
    EmptyE = sparse.coo_matrix((emptyE_data,(emptyE_rows,emptyE_cols)), shape=(row_ctr_e, total_vars))
    EmptyF = sparse.coo_matrix((emptyF_data,(emptyF_rows,emptyF_cols)), shape=(row_ctr_e, total_vars))
    Q_empty = cp.vstack(Q_empty_list)

    # (c) Supply‐alight
    saE_rows, saE_cols, saE_data = [], [], []
    saF_rows, saF_cols, saF_data = [], [], []
    Q_supply_list = []
    row_ctr_s = 0
    for s in range(S):
        off_A = s*N_A
        off_E = S*N_A + s*N_EF
        off_F = S*N_A + S*N_EF + s*N_EF
        for t in range(T):
            for i in range(r):
                # λ·A
                saE_rows.append(row_ctr_s)
                saE_cols.append(off_A + t*r + i)
                saE_data.append(lambda_avg_all[s, t, i])
                # -μ·E
                for j in range(r):
                    if j!=i:
                        saE_rows.append(row_ctr_s)
                        saE_cols.append(off_E + idx(t, j, i))
                        saE_data.append(-mu_avg_all[s, t, j, i])
                # μ·F with Q scaling
                temp_Q = None
                for j in range(r):
                    saF_rows.append(row_ctr_s)
                    saF_cols.append(off_F + idx(t, j, i))
                    saF_data.append(mu_avg_all[s, t, j, i])
                    temp_Q = Qp[t, j, i] if temp_Q is None else temp_Q + Qp[t, j, i]
                Q_supply_list.append(temp_Q)
                row_ctr_s += 1
    SAE = sparse.coo_matrix((saE_data,(saE_rows,saE_cols)), shape=(row_ctr_s, total_vars))
    SAF = sparse.coo_matrix((saF_data,(saF_rows,saF_cols)), shape=(row_ctr_s, total_vars))
    Q_supply = cp.vstack(Q_supply_list)

    # (d) Unit‐mass
    um_rows, um_cols, um_data = [], [], []
    row_ctr_u = 0
    for s in range(S):
        for m in range(N_EF):
            um_rows += [row_ctr_u, row_ctr_u]
            um_cols += [e_off + s*N_EF + m, f_off + s*N_EF + m]
            um_data += [1.0, 1.0]
            row_ctr_u += 1
    UMat = sparse.coo_matrix((um_data,(um_rows,um_cols)), shape=(row_ctr_u, total_vars))
    b_um = np.ones((row_ctr_u, 1))

    # Stack decision variable
    z = cp.vstack([cp.reshape(A_flat,(S*N_A,1)), cp.reshape(E_flat,(S*N_EF,1)), cp.reshape(F_flat,(S*N_EF,1))])

    # Objective: maximize worst‐case average reward
    wk = []
    for s in range(S):
        lamP = (lambda_avg_all[s][:, :, None] * P_avg_all[s]).sum(axis=2)
        expr = cp.sum(cp.multiply(cp.reshape(A_flat[s],(T, r)), lamP)) / T
        wk.append(expr)

    # Constraints
    constraints = []
    constraints += [RideMat @ z == 0]
    constraints += [EmptyE @ z == cp.multiply(Q_empty, EmptyF @ z)]
    constraints += [SAE @ z == cp.multiply(Q_supply, SAF @ z)]
    constraints += [UMat @ z <= b_um]
    constraints += [z[:S*N_A] <= 1]
    constraints += [eta <= cp.vstack(wk)]

    prob = cp.Problem(cp.Maximize(eta), constraints)
    return CvxpyLayer(prob, parameters=[Qp], variables=[eta])

#---------------------------------------
# 3) Instantiate and train
#---------------------------------------
layer = build_robust_routing_cvxpy(lambda_avg_all, mu_avg_all, P_avg_all)
opt = torch.optim.Adam([raw_Q], lr=1e-3)
# scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.1)

loss_history = []
for it in range(500):
    opt.zero_grad()
    eta_val, = layer(row_softmax(raw_Q), solver_args={"verbose": False,
                                                      # "solve_method": "ECOS"
                                                      })
    loss = -eta_val
    loss.backward()
    opt.step()
    # scheduler.step()
    print(f"iter {it}: η={eta_val.item():.8e}")
    loss_history.append(eta_val.item())
    # print(f"Q norm {torch.linalg.norm(raw_Q).item():.8f}")

# Function to plot the loss
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg' if TkAgg isn't available
import matplotlib.pyplot as plt
def plot_loss(loss_list, title="Training Loss"):
    plt.figure(figsize=(8, 4))
    plt.plot(loss_list, label='Loss', color='blue', linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.yscale('log')  # Optional: makes small values (like 1e-4) easier to visualize
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Example usage after training:
plot_loss(loss_history)