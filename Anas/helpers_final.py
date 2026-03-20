import jax
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt

from classes_final.py import (
    RBM,
    FFN,
    CNN,
    NeuralQuantumState,
    Sampler,
    TFIM,
    Observables,
    AdamOptimizer,
    SROptimizer,
) 

def exact_tfim_ground_energy(L, J, g):
    """
    Returns TOTAL ground state energy (not per site)
    """
    dim = 2 ** L

    states = ((jnp.arange(dim)[:, None] >> jnp.arange(L)) & 1)
    states = 2 * states - 1
    states = states.astype(jnp.int32)

    H = jnp.zeros((dim, dim), dtype=jnp.float32)

    diag = -J * jnp.sum(states * jnp.roll(states, -1, axis=1), axis=1)
    H = H.at[jnp.arange(dim), jnp.arange(dim)].set(diag)

    for i in range(L):
        flipped = states.at[:, i].set(-states[:, i])
        bits = ((flipped + 1) // 2).astype(jnp.int32)
        idx = jnp.sum(bits * (2 ** jnp.arange(L)), axis=1)
        H = H.at[jnp.arange(dim), idx].add(-g)

    evals = jnp.linalg.eigvalsh(H)
    return evals[0]


def build_model(model_name, L, key):
    if model_name == "RBM":
        hidden=20
        arch = RBM(L, hidden=hidden)
        model_info = {"hidden": hidden}

    elif model_name == "FFN":
        hidden_layers = [32, 32]
        arch = FFN(L, hidden_layers)
        model_info = {"hidden_layers": hidden_layers}

    elif model_name == "CNN":
        channels=16
        kernel=3
        arch = CNN(L, channels=channels, kernel=kernel)
        model_info = {"channels": channels, "kernel": kernel}

    else:
        raise ValueError(f"Unknown model: {model_name}")

    params = arch.init_params(key)
    return arch, params, model_info


def run_model(MODEL):
    L = 10
    J = 1.0
    g = 0.5

    n_steps = 70

    nchains = 32
    nsamples_per_chain = 16
    neq = 50
    nskip = 2

    adam_lr = 1e-2
    sr_lr = 0.05
    sr_diag_shift = 1e-2

    entropy_pairings = 32
    entropy_subsystem_size = L // 2

    key = random.PRNGKey(0)

    key, k = random.split(key)
    arch, params0, model_info = build_model(MODEL, L, k)

    print("\n" + "=" * 50)
    print("MODEL CONFIGURATION")
    print("=" * 50)
    print(f"Model type: {MODEL}")
    for kk, vv in model_info.items():
        print(f"  {kk}: {vv}")

    print("\nSystem:")
    print(f"  L = {L}")
    print(f"  J = {J}")
    print(f"  g = {g}")

    print("\nSampler:")
    print(f"  nchains = {nchains}")
    print(f"  nsamples_per_chain = {nsamples_per_chain}")
    print(f"  neq = {neq}")
    print(f"  nskip = {nskip}")

    print("\nOptimizers:")
    print(f"  Adam lr = {adam_lr}")
    print(f"  SR lr = {sr_lr}")
    print(f"  SR diag_shift = {sr_diag_shift}")

    print("\nEntropy:")
    print(f"  subsystem_size = {entropy_subsystem_size}")
    print(f"  n_pairings = {entropy_pairings}")
    print("=" * 50)

    params_adam = jax.tree_util.tree_map(lambda x: x.copy(), params0)
    params_sr = jax.tree_util.tree_map(lambda x: x.copy(), params0)

    wf_adam = NeuralQuantumState(arch, params_adam, L)
    wf_sr = NeuralQuantumState(arch, params_sr, L)

    obs_adam = Observables(wf_adam)
    obs_sr = Observables(wf_sr)

    sampler_adam = Sampler(nchains, nsamples_per_chain, neq, nskip, wf_adam)
    sampler_sr = Sampler(nchains, nsamples_per_chain, neq, nskip, wf_sr)

    ham_adam = TFIM(wf_adam, J, g)
    ham_sr = TFIM(wf_sr, J, g)

    adam = AdamOptimizer(wf_adam, ham_adam, sampler_adam, lr=adam_lr)
    sr = SROptimizer(wf_sr, ham_sr, sampler_sr, lr=sr_lr, diag_shift=sr_diag_shift)

    key, k1, k2, k3, k4 = random.split(key, 5)

    E_adam = []
    S_adam = []

    E_sr = []
    S_sr = []

    print(f"\n===== {MODEL} =====")

    for i in range(n_steps):
        # Adam step
        k1, sub1 = random.split(k1)
        params_adam, E_a, samples_a = adam.step(sub1, params_adam)

        k3, obs_key_a = random.split(k3)
        S_a = obs_adam.renyi2_entropy_swap(
            params_adam,
            samples_a,
            key=obs_key_a,
            subsystem_size=entropy_subsystem_size,
            n_pairings=entropy_pairings,
        )

        E_adam.append(E_a)
        S_adam.append(S_a)

        # SR step
        k2, sub2 = random.split(k2)
        params_sr, E_s, samples_s = sr.step(sub2, params_sr)

        k4, obs_key_s = random.split(k4)
        S_s = obs_sr.renyi2_entropy_swap(
            params_sr,
            samples_s,
            key=obs_key_s,
            subsystem_size=entropy_subsystem_size,
            n_pairings=entropy_pairings,
        )

        E_sr.append(E_s)
        S_sr.append(S_s)

        print(
            f"step {i:3d} | "
            f"Adam: E = {float(E_a): .6f}, S2 = {float(S_a): .6f} | "
            f"SR: E = {float(E_s): .6f}, S2 = {float(S_s): .6f}"
        )
    # -------------------------
    # Results
    # -------------------------
    print("\nFinal results:")
    print(f"{MODEL} + Adam final energy = {float(E_adam[-1]): .8f}")
    print(f"{MODEL} + SR   final energy = {float(E_sr[-1]): .8f}")

    if L <= 14:
        E0_exact = exact_tfim_ground_energy(L, J, g)
        print(f"Exact ground energy = {float(E0_exact): .8f}")
    return (
        jnp.array(E_adam),
        jnp.array(S_adam),
        jnp.array(E_sr),
        jnp.array(S_sr),
    )


def main():
    models = ["CNN"]#,"RBM" , "FFN" ]

    for m in models:
        E_a, S_a, E_s, S_s = run_model(m)

        plt.figure(figsize=(8, 5))
        plt.plot(E_a, label="Adam")
        plt.plot(E_s, label="SR")
        plt.title(f"{m} Total Energy")
        plt.xlabel("Step")
        plt.ylabel("Energy")
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8, 5))
        plt.plot(S_a, label="Adam")
        plt.plot(S_s, label="SR")
        plt.title(f"{m} Rényi-2 Entropy")
        plt.xlabel("Step")
        plt.ylabel("S2")
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()