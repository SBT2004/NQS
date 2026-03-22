from jax import random
import jax.numpy as jnp

from experiments import build_model
from models import NeuralQuantumState
from sampler import Sampler
from hamiltonians import TFIM
from observables import Observables
from ED import (
    exact_tfim_ground_energy,
    exact_energy_variance,
    exact_magnetization_z,
    exact_abs_magnetization_z,
    exact_spin_spin_correlation,
    exact_correlation_profile,
)


def rebuild_trained_state(
    result,
    optimizer="SR",
    sample_seed=1234,
    nchains=64,
    nsamples_per_chain=32,
    neq=100,
    nskip=4,
):
    model_name = result["model"]
    model_info = result["model_info"]
    params_dict = result["params"]

    L = params_dict["L"]
    J = params_dict["J"]
    g = params_dict["g"]
    seed = params_dict["seed"]

    arch_key = random.PRNGKey(seed + 999)

    arch, _, _ = build_model(
        model_name=model_name,
        L=L,
        key=arch_key,
        hidden=model_info.get("hidden", 20),
        hidden_layers=tuple(model_info.get("hidden_layers", (32, 32))),
        channels=model_info.get("channels", 16),
        kernel=model_info.get("kernel", 3),
        n_conv_layers=model_info.get("n_conv_layers", 1),
    )

    if optimizer.upper() == "SR":
        trained_params = result["final_params_sr"]
    else:
        trained_params = result["final_params_adam"]

    wf = NeuralQuantumState(arch, trained_params, L)
    sampler = Sampler(
        nchains=nchains,
        nsamples_per_chain=nsamples_per_chain,
        neq=neq,
        nskip=nskip,
        wavefunction=wf,
    )
    ham = TFIM(wf, J, g)
    obs = Observables(wf)

    sample_key = random.PRNGKey(sample_seed)
    samples = sampler.sample_chain(sample_key, trained_params)

    return {
        "architecture": arch,
        "params": trained_params,
        "wavefunction": wf,
        "sampler": sampler,
        "hamiltonian": ham,
        "observables": obs,
        "samples": samples,
        "L": L,
        "J": J,
        "g": g,
    }


def benchmark_observables(
    result,
    optimizer="SR",
    sample_seed=1234,
    nchains=64,
    nsamples_per_chain=32,
    neq=100,
    nskip=4,
    r_values=(1, 2, 3, 4),
):
    state = rebuild_trained_state(
        result=result,
        optimizer=optimizer,
        sample_seed=sample_seed,
        nchains=nchains,
        nsamples_per_chain=nsamples_per_chain,
        neq=neq,
        nskip=nskip,
    )

    wf = state["wavefunction"]
    params = state["params"]
    ham = state["hamiltonian"]
    obs = state["observables"]
    samples = state["samples"]
    L = state["L"]
    J = state["J"]
    g = state["g"]

    local_energies = ham.energy(params, samples)
    energy_mean = jnp.mean(local_energies)
    energy_var = obs.energy_variance(ham, params, samples)
    mz = obs.magnetization_z(samples)
    abs_mz = obs.abs_magnetization_z(samples)
    corr_profile = obs.correlation_profile(samples, r_values)

    exact = {
        "energy": exact_tfim_ground_energy(L, J, g),
        "energy_variance": exact_energy_variance(L, J, g),
        "magnetization_z": exact_magnetization_z(L, J, g),
        "abs_magnetization_z": exact_abs_magnetization_z(L, J, g),
        "correlation_profile": exact_correlation_profile(L, J, g, r_values),
    }

    nqs = {
        "energy": energy_mean,
        "energy_variance": energy_var,
        "magnetization_z": mz,
        "abs_magnetization_z": abs_mz,
        "correlation_profile": corr_profile,
    }

    return {
        "optimizer": optimizer,
        "system": {"L": L, "J": J, "g": g},
        "nqs": nqs,
        "exact": exact,
    }