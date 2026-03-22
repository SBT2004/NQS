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

    return {
        "model": model_name,
        "L": L,
        "seed": seed,
        "arch": arch,
        "params": params,
        "wavefunction": wf,
        "model_info": model_info,
        "n_parameters": count_parameters(params),
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
    return sampler.sample_chain(key, params)


def disorder_averaged_entropy_profile(
    model_name,
    *,
    L=10,
    seeds=range(10),
    subsystem_sizes=None,
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
    """
    Disorder-averaged Renyi-2 entropy profile S2(L_A) for one architecture.
    Disorder averaging is done over random network initializations.
    """
    if subsystem_sizes is None:
        subsystem_sizes = list(range(1, L // 2 + 1))

    subsystem_sizes = [int(x) for x in subsystem_sizes]
    seeds = list(seeds)

    profile_runs = []

    for seed in seeds:
        state = build_random_state(
            model_name=model_name,
            L=L,
            seed=int(seed),
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
            seed=int(10000 + seed),
        )

        obs = Observables(wf)
        profile = obs.entropy_profile(
            params=params,
            samples=samples,
            key=random.PRNGKey(int(20000 + seed)),
            subsystem_sizes=subsystem_sizes,
            n_pairings=entropy_pairings,
        )

        profile_runs.append(
            jnp.array([profile[s] for s in subsystem_sizes], dtype=jnp.float32)
        )

    profile_runs = jnp.stack(profile_runs, axis=0)

    return {
        "model": model_name,
        "L": L,
        "subsystem_sizes": subsystem_sizes,
        "seeds": seeds,
        "n_runs": len(seeds),
        "mean": jnp.mean(profile_runs, axis=0),
        "std": jnp.std(profile_runs, axis=0),
        "all_runs": profile_runs,
    }


def compare_architectures_entropy_scaling(
    *,
    L=10,
    seeds=range(10),
    subsystem_sizes=None,
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
    """
    Returns disorder-averaged S2(L_A) curves for RBM, FFN, CNN.
    """
    results = {}

    results["RBM"] = disorder_averaged_entropy_profile(
        "RBM",
        L=L,
        seeds=seeds,
        subsystem_sizes=subsystem_sizes,
        entropy_pairings=entropy_pairings,
        nchains=nchains,
        nsamples_per_chain=nsamples_per_chain,
        neq=neq,
        nskip=nskip,
        hidden=rbm_hidden,
    )

    results["FFN"] = disorder_averaged_entropy_profile(
        "FFN",
        L=L,
        seeds=seeds,
        subsystem_sizes=subsystem_sizes,
        entropy_pairings=entropy_pairings,
        nchains=nchains,
        nsamples_per_chain=nsamples_per_chain,
        neq=neq,
        nskip=nskip,
        hidden_layers=ffn_hidden_layers,
    )

    results["CNN"] = disorder_averaged_entropy_profile(
        "CNN",
        L=L,
        seeds=seeds,
        subsystem_sizes=subsystem_sizes,
        entropy_pairings=entropy_pairings,
        nchains=nchains,
        nsamples_per_chain=nsamples_per_chain,
        neq=neq,
        nskip=nskip,
        channels=cnn_channels,
        kernel=cnn_kernel,
        n_conv_layers=cnn_n_conv_layers,
    )

    return results

def compare_swap_vs_exact_for_one_model(
    model_name,
    *,
    L=10,
    seeds=range(10),
    subsystem_sizes=None,
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
    """
    For the same random states, compare:
      - swap-estimated S2
      - exact S2 from full basis enumeration

    Only use for small L.
    """
    if subsystem_sizes is None:
        subsystem_sizes = list(range(1, L // 2 + 1))

    subsystem_sizes = [int(x) for x in subsystem_sizes]
    seeds = list(seeds)

    swap_runs = []
    exact_runs = []

    for seed in seeds:
        state = build_random_state(
            model_name=model_name,
            L=L,
            seed=int(seed),
            hidden=hidden,
            hidden_layers=hidden_layers,
            channels=channels,
            kernel=kernel,
            n_conv_layers=n_conv_layers,
        )

        wf = state["wavefunction"]
        params = state["params"]
        obs = Observables(wf)

        samples = sample_random_state(
            wf,
            params,
            nchains=nchains,
            nsamples_per_chain=nsamples_per_chain,
            neq=neq,
            nskip=nskip,
            seed=int(10000 + seed),
        )

        swap_vals = []
        exact_vals = []

        for LA in subsystem_sizes:
            s2_swap = obs.renyi2_entropy_swap(
                params=params,
                samples=samples,
                key=random.PRNGKey(int(20000 + 100 * seed + LA)),
                subsystem_size=LA,
                n_pairings=entropy_pairings,
            )
            s2_exact = obs.renyi2_entropy_exact(
                params=params,
                subsystem_size=LA,
            )

            swap_vals.append(s2_swap)
            exact_vals.append(s2_exact)

        swap_runs.append(jnp.array(swap_vals, dtype=jnp.float32))
        exact_runs.append(jnp.array(exact_vals, dtype=jnp.float32))

    swap_runs = jnp.stack(swap_runs, axis=0)
    exact_runs = jnp.stack(exact_runs, axis=0)

    return {
        "model": model_name,
        "L": L,
        "subsystem_sizes": subsystem_sizes,
        "seeds": seeds,
        "swap_mean": jnp.mean(swap_runs, axis=0),
        "swap_std": jnp.std(swap_runs, axis=0),
        "exact_mean": jnp.mean(exact_runs, axis=0),
        "exact_std": jnp.std(exact_runs, axis=0),
        "swap_all_runs": swap_runs,
        "exact_all_runs": exact_runs,
    }


def compare_swap_vs_exact_cnn_depths(
    *,
    L=10,
    seeds=range(10),
    subsystem_sizes=None,
    cnn_channels=16,
    cnn_kernel=3,
    depth_values=(1, 3, 5),
    entropy_pairings=16,
    nchains=32,
    nsamples_per_chain=16,
    neq=50,
    nskip=2,
):
    """
    Compare swap vs exact S2 for CNNs with different depths.
    """
    results = {}

    for depth in depth_values:
        results[f"CNN_{depth}layers"] = compare_swap_vs_exact_for_one_model(
            "CNN",
            L=L,
            seeds=seeds,
            subsystem_sizes=subsystem_sizes,
            channels=cnn_channels,
            kernel=cnn_kernel,
            n_conv_layers=depth,
            entropy_pairings=entropy_pairings,
            nchains=nchains,
            nsamples_per_chain=nsamples_per_chain,
            neq=neq,
            nskip=nskip,
        )

    return results

def disorder_averaged_entropy_vs_parameter_count(
    model_name,
    size_values,
    *,
    L=10,
    seeds=range(10),
    subsystem_size=None,
    entropy_pairings=16,
    nchains=32,
    nsamples_per_chain=16,
    neq=50,
    nskip=2,
    kernel=3,
):
    """
    Disorder-averaged Renyi-2 entropy at fixed subsystem size,
    scanned versus model size / parameter count.

    For RBM:
        size_values = [4, 8, 12, ...] meaning hidden units

    For FFN:
        size_values = [(8,), (16,), (32,), (32,32), ...] meaning hidden_layers

    For CNN:
        size_values = [(8,1), (16,1), (16,3), ...]
        interpreted as (channels, n_conv_layers)
    """
    if subsystem_size is None:
        subsystem_size = L // 2

    seeds = list(seeds)
    results = []

    for size_val in size_values:
        kwargs = {}

        if model_name == "RBM":
            kwargs["hidden"] = int(size_val)
            label = f"hidden={int(size_val)}"

        elif model_name == "FFN":
            kwargs["hidden_layers"] = tuple(size_val)
            label = f"layers={tuple(size_val)}"

        elif model_name == "CNN":
            channels, n_conv_layers = size_val
            kwargs["channels"] = int(channels)
            kwargs["kernel"] = kernel
            kwargs["n_conv_layers"] = int(n_conv_layers)
            label = f"channels={int(channels)}, layers={int(n_conv_layers)}"

        else:
            raise ValueError(f"Unknown model: {model_name}")

        s2_runs = []
        n_params_ref = None

        for seed in seeds:
            state = build_random_state(
                model_name=model_name,
                L=L,
                seed=int(seed),
                **kwargs,
            )

            wf = state["wavefunction"]
            params = state["params"]
            n_params_ref = state["n_parameters"]

            samples = sample_random_state(
                wf,
                params,
                nchains=nchains,
                nsamples_per_chain=nsamples_per_chain,
                neq=neq,
                nskip=nskip,
                seed=int(10000 + seed),
            )

            obs = Observables(wf)
            s2 = obs.renyi2_entropy_swap(
                params=params,
                samples=samples,
                key=random.PRNGKey(int(20000 + seed)),
                subsystem_size=subsystem_size,
                n_pairings=entropy_pairings,
            )

            s2_runs.append(s2)

        s2_runs = jnp.array(s2_runs, dtype=jnp.float32)

        results.append({
            "model": model_name,
            "size_value": size_val,
            "label": label,
            "n_parameters": int(n_params_ref),
            "subsystem_size": int(subsystem_size),
            "S2_mean": jnp.mean(s2_runs),
            "S2_std": jnp.std(s2_runs),
            "all_runs": s2_runs,
        })

    return results

def compare_architectures_entropy_vs_parameter_count(
    *,
    L=10,
    seeds=range(10),
    subsystem_size=None,
    entropy_pairings=16,
    nchains=32,
    nsamples_per_chain=16,
    neq=50,
    nskip=2,
    rbm_size_values=(4, 8, 12, 16, 20, 24),
    ffn_size_values=((8,), (16,), (32,), (32, 32), (64, 64)),
    cnn_size_values=((4, 1), (8, 1), (16, 1), (16, 2), (16, 3)),
    cnn_kernel=3,
    ):
    """
    Returns disorder-averaged S2 vs parameter-count scans for RBM, FFN, CNN.
    """
    results = {}

    results["RBM"] = disorder_averaged_entropy_vs_parameter_count(
        "RBM",
        size_values=rbm_size_values,
        L=L,
        seeds=seeds,
        subsystem_size=subsystem_size,
        entropy_pairings=entropy_pairings,
        nchains=nchains,
        nsamples_per_chain=nsamples_per_chain,
        neq=neq,
        nskip=nskip,
    )

    results["FFN"] = disorder_averaged_entropy_vs_parameter_count(
        "FFN",
        size_values=ffn_size_values,
        L=L,
        seeds=seeds,
        subsystem_size=subsystem_size,
        entropy_pairings=entropy_pairings,
        nchains=nchains,
        nsamples_per_chain=nsamples_per_chain,
        neq=neq,
        nskip=nskip,
    )

    results["CNN"] = disorder_averaged_entropy_vs_parameter_count(
        "CNN",
        size_values=cnn_size_values,
        L=L,
        seeds=seeds,
        subsystem_size=subsystem_size,
        entropy_pairings=entropy_pairings,
        nchains=nchains,
        nsamples_per_chain=nsamples_per_chain,
        neq=neq,
        nskip=nskip,
        kernel=cnn_kernel,
    )

    return results