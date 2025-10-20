
import numpy as np
import matplotlib.pyplot as plt

def choquet_integral_signed(f, mu_singletons, mu_pair):
    f = np.asarray(f, dtype=float)
    mu1, mu2 = mu_singletons
    idx = np.argsort(f)
    f_sorted = f[idx]
    mu_A1 = mu_pair
    largest = idx[1]
    mu_A2 = mu2 if largest == 1 else mu1
    return f_sorted[0] * mu_A1 + (f_sorted[1] - f_sorted[0]) * mu_A2

def decision_value(x, params):
    a0, a1 = params["a"]
    b0, b1 = params["b"]
    mu12, mu1, mu2 = params["mu"]
    B = params["B"]
    g0 = a0 + b0 * x[..., 0]
    g1 = a1 + b1 * x[..., 1]
    cval = np.vectorize(lambda u, v: choquet_integral_signed((u, v), (mu1, mu2), mu12))
    y = cval(g0, g1)
    return y - B

def generate_and_plot(params, N=200, seed=0, savepath="fig_replication.png", title="Fig 3 Replication"):
    rng = np.random.default_rng(seed)
    X = rng.random((N, 2))
    dv = decision_value(X, params)
    A_mask = dv > 0.0
    Aprime_mask = ~A_mask
    nA = int(A_mask.sum())
    nAprime = int(Aprime_mask.sum())
    plt.figure(figsize=(6, 5), dpi=150)
    plt.scatter(X[A_mask, 0], X[A_mask, 1], s=12, label="Class A", alpha=0.9)
    plt.scatter(X[Aprime_mask, 0], X[Aprime_mask, 1], s=12, marker="x", label="Class A'")
    gx = np.linspace(0, 1, 300)
    gy = np.linspace(0, 1, 300)
    GX, GY = np.meshgrid(gx, gy)
    grid = np.stack([GX, GY], axis=-1)
    Z = decision_value(grid, params)
    CS = plt.contour(GX, GY, Z, levels=[0.0], linewidths=1.0)
    plt.clabel(CS, inline=True, fontsize=8, fmt={0.0: "Boundary"})
    plt.title(f"{title}\nCounts → A: {nA}, A': {nAprime}")
    plt.xlabel("x1"); plt.ylabel("x2"); plt.legend(loc="best"); plt.tight_layout()
    plt.savefig(savepath, bbox_inches="tight"); plt.close()
    return nA, nAprime, savepath

params_3a = dict(mu=(0.1389, 0.1802, 0.5460), B=0.0917, a=(0.0, 0.0), b=(1.0, 1.0))
params_3b = dict(mu=(0.3830, 0.6683, 0.5713), B=0.2633, a=(0.4420, 0.7021), b=(0.3614, -0.154))

if __name__ == "__main__":
    which = "3b"  # change to "3b" to replicate Fig 3(b)
    if which == "3a":
        nA, nAprime, path = generate_and_plot(params_3a, N=200, seed=1, savepath="fig3a_replication.png",
                                              title="Replication of Fig 3(a) parameters")
        print("Replicated Fig 3(a) →", nA, nAprime, path)
    else:
        nA, nAprime, path = generate_and_plot(params_3b, N=200, seed=2, savepath="fig3b_replication.png",
                                              title="Replication of Fig 3(b) parameters")
        print("Replicated Fig 3(b) →", nA, nAprime, path)
