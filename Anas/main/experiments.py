import jax
import jax.numpy as jnp
from jax import random

from models import RBM, FFN, CNN, NeuralQuantumState
from sampler import Sampler
from hamiltonians import TFIM
from observables import Observables
from optimizers import AdamOptimizer, SROptimizer
from ED import exact_tfim_ground_energy


def build_model(
    model_name,
    L,
    key,
    hidden=20,
    hidden_layers=(32, 32),
    channels=16,
    kernel=3,
    n_conv_layers=1,
):
    if model_name == "RBM":
        arch = RBM(L, hidden=hidden)
        model_info = {"hidden": hidden}

    elif model_name == "FFN":
        hidden_layers = list(hidden_layers)
        arch = FFN(L, hidden_layers=hidden_layers)
        model_info = {"hidden_layers": hidden_layers}

    elif model_name == "CNN":
        arch = CNN(L, channels=channels, kernel=kernel, n_conv_layers=n_conv_layers)
        model_info = {
            "channels": channels,
            "kernel": kernel,
            "n_conv_layers": n_conv_layers,
        }

    else:
        raise ValueError(f"Unknown model: {model_name}")

    params = arch.init_params(key)
    return arch, params, model_info


def run_model(
    model_name,
    *,
    L=10,
    J=1.0,
    g=0.5,
    n_steps=70,
    nchains=32,
    nsamples_per_chain=16,
    neq=50,
    nskip=2,
    adam_lr=1e-2,
    sr_lr=0.05,
    sr_diag_shift=1e-2,
    entropy_pairings=16,
    entropy_subsystem_size=None,
    seed=0,
    hidden=20,
    hidden_layers=(32, 32),
    channels=16,
    kernel=3,
    n_conv_layers=1,
    verbose=True,
):
    if entropy_subsystem_size is None:
        entropy_subsystem_size = L // 2

    key = random.PRNGKey(seed)
    key, k = random.split(key)

    arch, params0, model_info = build_model(
        model_name=model_name,
        L=L,
        key=k,
        hidden=hidden,
        hidden_layers=hidden_layers,
        channels=channels,
        kernel=kernel,
        n_conv_layers=n_conv_layers,
    )

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

    E_adam, S_adam = [], []
    E_sr, S_sr = [], []

    if verbose:
        print(f"\n===== {model_name} | L={L}, J={J}, g={g} =====")

    for i in range(n_steps):
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

        if verbose:
            print(
                f"step {i:3d} | "
                f"Adam: E={float(E_a): .6f}, S2={float(S_a): .6f} | "
                f"SR: E={float(E_s): .6f}, S2={float(S_s): .6f}"
            )

    E_adam = jnp.array(E_adam)
    S_adam = jnp.array(S_adam)
    E_sr = jnp.array(E_sr)
    S_sr = jnp.array(S_sr)

    exact_energy = None
    if L <= 14:
        exact_energy = exact_tfim_ground_energy(L, J, g)

    return {
        "model": model_name,
        "model_info": model_info,
        "params": {
            "L": L,
            "J": J,
            "g": g,
            "n_steps": n_steps,
            "nchains": nchains,
            "nsamples_per_chain": nsamples_per_chain,
            "neq": neq,
            "nskip": nskip,
            "adam_lr": adam_lr,
            "sr_lr": sr_lr,
            "sr_diag_shift": sr_diag_shift,
            "entropy_pairings": entropy_pairings,
            "entropy_subsystem_size": entropy_subsystem_size,
            "seed": seed,
        },
        "E_adam": E_adam,
        "S_adam": S_adam,
        "E_sr": E_sr,
        "S_sr": S_sr,
        "exact_energy": exact_energy,
        "final_params_adam": params_adam,
        "final_params_sr": params_sr,
    }


def run_architecture_comparison(models=("RBM", "FFN", "CNN"), **kwargs):
    results = {}
    for m in models:
        results[m] = run_model(m, **kwargs)
    return results


def run_g_scan(
    model_name,
    g_values,
    *,
    use_optimizer="SR",
    **kwargs,
):
    results = []
    for g in g_values:
        res = run_model(model_name, g=float(g), **kwargs)
        if use_optimizer.upper() == "SR":
            final_energy = float(res["E_sr"][-1])
            final_entropy = float(res["S_sr"][-1])
        else:
            final_energy = float(res["E_adam"][-1])
            final_entropy = float(res["S_adam"][-1])

        exact_energy = None if res["exact_energy"] is None else float(res["exact_energy"])

        results.append({
            "g": float(g),
            "final_energy": final_energy,
            "final_entropy": final_entropy,
            "exact_energy": exact_energy,
            "full_result": res,
        })
    return results


def run_L_scan(
    model_name,
    L_values,
    *,
    use_optimizer="SR",
    **kwargs,
):
    results = []
    for L in L_values:
        res = run_model(model_name, L=int(L), **kwargs)
        if use_optimizer.upper() == "SR":
            final_energy = float(res["E_sr"][-1])
            final_entropy = float(res["S_sr"][-1])
        else:
            final_energy = float(res["E_adam"][-1])
            final_entropy = float(res["S_adam"][-1])

        exact_energy = None if res["exact_energy"] is None else float(res["exact_energy"])

        results.append({
            "L": int(L),
            "final_energy": final_energy,
            "final_entropy": final_entropy,
            "exact_energy": exact_energy,
            "full_result": res,
        })
    return results


def run_entropy_vs_subsystem_size(
    model_name,
    subsystem_sizes,
    *,
    optimizer="SR",
    L=10,
    J=1.0,
    g=0.5,
    entropy_pairings=16,
    seed=0,
    **kwargs,
):
    res = run_model(
        model_name,
        L=L,
        J=J,
        g=g,
        entropy_pairings=entropy_pairings,
        seed=seed,
        **kwargs,
    )

    key = random.PRNGKey(seed + 12345)

    if optimizer.upper() == "SR":
        params = res["final_params_sr"]
    else:
        params = res["final_params_adam"]

    arch_key = random.PRNGKey(seed + 999)
    arch, params0, _ = build_model(
        model_name=model_name,
        L=L,
        key=arch_key,
        hidden=kwargs.get("hidden", 20),
        hidden_layers=kwargs.get("hidden_layers", (32, 32)),
        channels=kwargs.get("channels", 16),
        kernel=kwargs.get("kernel", 3),
        n_conv_layers=kwargs.get("n_conv_layers", 1),
    )

    wf = NeuralQuantumState(arch, params0, L)
    obs = Observables(wf)
    sampler = Sampler(
        kwargs.get("nchains", 32),
        kwargs.get("nsamples_per_chain", 16),
        kwargs.get("neq", 50),
        kwargs.get("nskip", 2),
        wf,
    )

    sample_key = random.PRNGKey(seed + 54321)
    samples = sampler.sample_chain(sample_key, params)

    profile = obs.entropy_profile(
        params=params,
        samples=samples,
        key=key,
        subsystem_sizes=subsystem_sizes,
        n_pairings=entropy_pairings,
    )

    return {
        "training_result": res,
        "subsystem_sizes": list(subsystem_sizes),
        "entropy_profile": profile,
        "optimizer": optimizer,
    }