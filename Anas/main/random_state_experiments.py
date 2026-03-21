import jax
import jax.numpy as jnp
from jax import random

from models import RBM, FFN, CNN, NeuralQuantumState, count_parameters
from sampler import Sampler
from observables import Observables


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


def build_random_state(
    model_name,
    *,
    L=10,
    seed=0,
    hidden=20,
    hidden_layers=(32, 32),
    channels=16,
    kernel=3,
    n_conv_layers=1,
):
    key = random.PRNGKey(seed)
    arch, params, model_info = build_model(
        model_name=model_name,
        L=L,
        key=key,
        hidden=hidden,
        hidden_layers=hidden_layers,
        channels=channels,
        kernel=kernel,
        n_conv_layers=n_conv_layers,
    )
    wf = NeuralQuantumState(arch, params, L)
    n_params = count_parameters(params)

    return {
        "model": model_name,
        "L": L,
        "seed": seed,
        "arch": arch,
        "params": params,
        "wavefunction": wf,
        "model_info": model_info,
        "n_parameters": n_params,
    }


def sample_random_state(
    wavefunction,
    params,
    *,
    nchains=32,
    nsamples_per_chain=16,
    neq=50,
    nskip=2,
    seed=123,
):
    sampler = Sampler(
        nchains=nchains,
        nsamples_per_chain=nsamples_per_chain,
        neq=neq,
        nskip=nskip,
        wavefunction=wavefunction,
    )
    key = random.PRNGKey(seed)
    samples = sampler.sample_chain(key, params)
    return samples


def measure_random_state_observables(
    model_name,
    *,
    L=10,
    model_seed=0,
    sample_seed=123,
    entropy_seed=999,
    subsystem_size=None,
    entropy_pairings=16,
    nchains=32,
    nsamples_per_chain=16,
    neq=50,
    nskip=2,
    hidden=20,
    hidden_layers=(32, 32),
    channels=16,
    kernel=3,
    n_conv_layers=1,
):
    state = build_random_state(
        model_name=model_name,
        L=L,
        seed=model_seed,
        hidden=hidden,
        hidden_layers=hidden_layers,
        channels=channels,
        kernel=kernel,
        n_conv_layers=n_conv_layers,
    )

    wf = state["wavefunction"]
    params = state["params"]

    samples = sample_random_state(
        wf,
        params,
        nchains=nchains,
        nsamples_per_chain=nsamples_per_chain,
        neq=neq,
        nskip=nskip,
        seed=sample_seed,
    )

    obs = Observables(wf)

    if subsystem_size is None:
        subsystem_size = L // 2

    S2 = obs.renyi2_entropy_swap(
        params=params,
        samples=samples,
        key=random.PRNGKey(entropy_seed),
        subsystem_size=subsystem_size,
        n_pairings=entropy_pairings,
    )

    mz = obs.magnetization_z(samples)
    abs_mz = obs.abs_magnetization_z(samples)

    corr_01 = obs.two_point_correlation_z(samples, 0, min(1, L - 1))
    conn_corr_01 = obs.connected_two_point_correlation_z(samples, 0, min(1, L - 1))

    return {
        "model": model_name,
        "L": L,
        "model_seed": model_seed,
        "sample_seed": sample_seed,
        "entropy_seed": entropy_seed,
        "model_info": state["model_info"],
        "n_parameters": state["n_parameters"],
        "samples": samples,
        "params": params,
        "wavefunction": wf,
        "S2": S2,
        "magnetization_z": mz,
        "abs_magnetization_z": abs_mz,
        "corr_01": corr_01,
        "connected_corr_01": conn_corr_01,
    }


def average_entropy_over_seeds(
    model_name,
    seeds,
    *,
    L=10,
    subsystem_size=None,
    entropy_pairings=16,
    nchains=32,
    nsamples_per_chain=16,
    neq=50,
    nskip=2,
    hidden=20,
    hidden_layers=(32, 32),
    channels=16,
    kernel=3,
    n_conv_layers=1,
):
    results = []

    for s in seeds:
        res = measure_random_state_observables(
            model_name=model_name,
            L=L,
            model_seed=int(s),
            sample_seed=int(1000 + s),
            entropy_seed=int(2000 + s),
            subsystem_size=subsystem_size,
            entropy_pairings=entropy_pairings,
            nchains=nchains,
            nsamples_per_chain=nsamples_per_chain,
            neq=neq,
            nskip=nskip,
            hidden=hidden,
            hidden_layers=hidden_layers,
            channels=channels,
            kernel=kernel,
            n_conv_layers=n_conv_layers,
        )
        results.append(res)

    S2_vals = jnp.array([r["S2"] for r in results], dtype=jnp.float32)
    mz_vals = jnp.array([r["magnetization_z"] for r in results], dtype=jnp.float32)
    abs_mz_vals = jnp.array([r["abs_magnetization_z"] for r in results], dtype=jnp.float32)

    return {
        "model": model_name,
        "L": L,
        "n_runs": len(seeds),
        "results": results,
        "S2_mean": jnp.mean(S2_vals),
        "S2_std": jnp.std(S2_vals),
        "mz_mean": jnp.mean(mz_vals),
        "mz_std": jnp.std(mz_vals),
        "abs_mz_mean": jnp.mean(abs_mz_vals),
        "abs_mz_std": jnp.std(abs_mz_vals),
    }


def scan_subsystem_sizes_random_state(
    model_name,
    subsystem_sizes,
    *,
    L=10,
    model_seed=0,
    sample_seed=123,
    entropy_seed=999,
    entropy_pairings=16,
    nchains=32,
    nsamples_per_chain=16,
    neq=50,
    nskip=2,
    hidden=20,
    hidden_layers=(32, 32),
    channels=16,
    kernel=3,
    n_conv_layers=1,
):
    state = build_random_state(
        model_name=model_name,
        L=L,
        seed=model_seed,
        hidden=hidden,
        hidden_layers=hidden_layers,
        channels=channels,
        kernel=kernel,
        n_conv_layers=n_conv_layers,
    )

    wf = state["wavefunction"]
    params = state["params"]

    samples = sample_random_state(
        wf,
        params,
        nchains=nchains,
        nsamples_per_chain=nsamples_per_chain,
        neq=neq,
        nskip=nskip,
        seed=sample_seed,
    )

    obs = Observables(wf)
    profile = obs.entropy_profile(
        params=params,
        samples=samples,
        key=random.PRNGKey(entropy_seed),
        subsystem_sizes=subsystem_sizes,
        n_pairings=entropy_pairings,
    )

    return {
        "model": model_name,
        "L": L,
        "model_info": state["model_info"],
        "n_parameters": state["n_parameters"],
        "subsystem_sizes": list(subsystem_sizes),
        "entropy_profile": profile,
    }


def scan_model_size_random_state(
    model_name,
    size_values,
    *,
    L=10,
    model_seed=0,
    sample_seed=123,
    entropy_seed=999,
    subsystem_size=None,
    entropy_pairings=16,
    nchains=32,
    nsamples_per_chain=16,
    neq=50,
    nskip=2,
):
    results = []

    for val in size_values:
        kwargs = {}
        if model_name == "RBM":
            kwargs["hidden"] = int(val)
        elif model_name == "FFN":
            kwargs["hidden_layers"] = tuple(val)
        elif model_name == "CNN":
            kwargs["channels"] = int(val)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        res = measure_random_state_observables(
            model_name=model_name,
            L=L,
            model_seed=model_seed,
            sample_seed=sample_seed,
            entropy_seed=entropy_seed,
            subsystem_size=subsystem_size,
            entropy_pairings=entropy_pairings,
            nchains=nchains,
            nsamples_per_chain=nsamples_per_chain,
            neq=neq,
            nskip=nskip,
            **kwargs,
        )

        results.append({
            "size_value": val,
            "n_parameters": res["n_parameters"],
            "S2": float(res["S2"]),
            "magnetization_z": float(res["magnetization_z"]),
            "abs_magnetization_z": float(res["abs_magnetization_z"]),
            "full_result": res,
        })

    return results


def compare_architectures_random_state(
    models=("RBM", "FFN", "CNN"),
    *,
    L=10,
    model_seed=0,
    sample_seed=123,
    entropy_seed=999,
    subsystem_size=None,
    entropy_pairings=16,
    nchains=32,
    nsamples_per_chain=16,
    neq=50,
    nskip=2,
    rbm_hidden=20,
    ffn_hidden_layers=(32, 32),
    cnn_channels=16,
    cnn_kernel=3,
    cnn_n_conv_layers=1,
):
    results = {}

    for model_name in models:
        kwargs = {}
        if model_name == "RBM":
            kwargs["hidden"] = rbm_hidden
        elif model_name == "FFN":
            kwargs["hidden_layers"] = ffn_hidden_layers
        elif model_name == "CNN":
            kwargs["channels"] = cnn_channels
            kwargs["kernel"] = cnn_kernel
            kwargs["n_conv_layers"] = cnn_n_conv_layers

        results[model_name] = measure_random_state_observables(
            model_name=model_name,
            L=L,
            model_seed=model_seed,
            sample_seed=sample_seed,
            entropy_seed=entropy_seed,
            subsystem_size=subsystem_size,
            entropy_pairings=entropy_pairings,
            nchains=nchains,
            nsamples_per_chain=nsamples_per_chain,
            neq=neq,
            nskip=nskip,
            **kwargs,
        )

    return results