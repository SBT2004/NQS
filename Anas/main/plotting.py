import matplotlib.pyplot as plt


def plot_training_curves(result, title_prefix=None):
    model = result["model"]
    exact_energy = result["exact_energy"]
    exact_entropy = result.get("exact_entropy", None)
    prefix = f"{title_prefix} - " if title_prefix else ""

    plt.figure(figsize=(8, 5))
    plt.plot(result["E_adam"], label="Adam")
    plt.plot(result["E_sr"], label="SR")
    if exact_energy is not None:
        plt.axhline(float(exact_energy), linestyle="--", label="Exact")
    plt.title(f"{prefix}{model} Total Energy")
    plt.xlabel("Step")
    plt.ylabel("Energy")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(result["S_adam"], label="Adam")
    plt.plot(result["S_sr"], label="SR")
    if exact_entropy is not None:
        plt.axhline(float(exact_entropy), linestyle="--", label="Exact")
    plt.title(f"{prefix}{model} Rényi-2 Entropy")
    plt.xlabel("Step")
    plt.ylabel("S2")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_g_scan(scan_results, title="Final quantities vs g"):
    g_vals = [r["g"] for r in scan_results]
    E_vals = [r["final_energy"] for r in scan_results]
    S_vals = [r["final_entropy"] for r in scan_results]
    E_exact = [r["exact_energy"] for r in scan_results]
    S_exact = [r.get("exact_entropy", None) for r in scan_results]

    plt.figure(figsize=(8, 5))
    plt.plot(g_vals, E_vals, marker="o", label="VMC final energy")
    if all(v is not None for v in E_exact):
        plt.plot(g_vals, E_exact, marker="s", linestyle="--", label="Exact energy")
    plt.title(title + " - Energy")
    plt.xlabel("g")
    plt.ylabel("Energy")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(g_vals, S_vals, marker="o", label="VMC final S2")
    if all(v is not None for v in S_exact):
        plt.plot(g_vals, S_exact, marker="s", linestyle="--", label="Exact S2")
    plt.title(title + " - Rényi-2 entropy")
    plt.xlabel("g")
    plt.ylabel("S2")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_g_scan_with_errorbars(scan_results, title="Final quantities vs g"):
    g_vals = [r["g"] for r in scan_results]

    E_mean = [r["energy_mean"] for r in scan_results]
    E_std = [r["energy_std"] for r in scan_results]
    E_exact = [r["exact_energy"] for r in scan_results]

    S_mean = [r["entropy_mean"] for r in scan_results]
    S_std = [r["entropy_std"] for r in scan_results]
    S_exact = [r["exact_entropy"] for r in scan_results]

    plt.figure(figsize=(8, 5))
    plt.errorbar(g_vals, E_mean, yerr=E_std, marker="o", capsize=4, label="VMC final energy")
    if all(v is not None for v in E_exact):
        plt.plot(g_vals, E_exact, marker="s", linestyle="--", label="Exact energy")
    plt.title(title + " - Energy")
    plt.xlabel("g")
    plt.ylabel("Energy")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.errorbar(g_vals, S_mean, yerr=S_std, marker="o", capsize=4, label="VMC final S2")
    if all(v is not None for v in S_exact):
        plt.plot(g_vals, S_exact, marker="s", linestyle="--", label="Exact S2")
    plt.title(title + " - Rényi-2 entropy")
    plt.xlabel("g")
    plt.ylabel("S2")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_L_scan(scan_results, title="Final quantities vs L"):
    L_vals = [r["L"] for r in scan_results]
    E_vals = [r["final_energy"] for r in scan_results]
    S_vals = [r["final_entropy"] for r in scan_results]
    E_exact = [r["exact_energy"] for r in scan_results]

    plt.figure(figsize=(8, 5))
    plt.plot(L_vals, E_vals, marker="o", label="VMC final energy")
    if all(v is not None for v in E_exact):
        plt.plot(L_vals, E_exact, marker="s", linestyle="--", label="Exact energy")
    plt.title(title + " - Energy")
    plt.xlabel("L")
    plt.ylabel("Energy")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(L_vals, S_vals, marker="o")
    plt.title(title + " - Rényi-2 entropy")
    plt.xlabel("L")
    plt.ylabel("S2")
    plt.tight_layout()
    plt.show()


def plot_entropy_profile(profile_result, title="Rényi-2 entropy vs subsystem size"):
    sizes = profile_result["subsystem_sizes"]
    vals = [float(profile_result["entropy_profile"][int(s)]) for s in sizes]

    plt.figure(figsize=(8, 5))
    plt.plot(sizes, vals, marker="o")
    plt.title(title)
    plt.xlabel("Subsystem size")
    plt.ylabel("S2")
    plt.tight_layout()
    plt.show()

# Plotting helpers for problem 2

def plot_random_architecture_comparison(results, title="Random state comparison"):
    """
    results: dict from compare_architectures_random_state
    """
    names = []
    S2 = []
    mz = []
    n_params = []

    for k, v in results.items():
        names.append(k)
        S2.append(float(v["S2"]))
        mz.append(float(v["abs_magnetization_z"]))
        n_params.append(v["n_parameters"])

    # Entropy
    plt.figure(figsize=(8, 5))
    plt.bar(names, S2)
    plt.ylabel("Rényi-2 entropy")
    plt.title(title + " (Entropy)")
    plt.tight_layout()
    plt.show()

    # Magnetization
    plt.figure(figsize=(8, 5))
    plt.bar(names, mz)
    plt.ylabel(r"$\langle |m_z| \rangle$")
    plt.title(title + " (Magnetization)")
    plt.tight_layout()
    plt.show()

    # Parameter count
    plt.figure(figsize=(8, 5))
    plt.bar(names, n_params)
    plt.ylabel("Number of parameters")
    plt.title(title + " (Model size)")
    plt.tight_layout()
    plt.show()


def plot_entropy_vs_subsystem(profile_result, title="Entropy vs subsystem size"):
    sizes = profile_result["subsystem_sizes"]
    vals = [float(profile_result["entropy_profile"][int(s)]) for s in sizes]

    plt.figure(figsize=(8, 5))
    plt.plot(sizes, vals, marker="o")
    plt.xlabel("Subsystem size")
    plt.ylabel("Rényi-2 entropy")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_parameter_scan(scan_results, title="Entropy vs model size"):
    sizes = [r["size_value"] for r in scan_results]
    S2 = [r["S2"] for r in scan_results]
    n_params = [r["n_parameters"] for r in scan_results]

    # vs hyperparameter
    plt.figure(figsize=(8, 5))
    plt.plot(sizes, S2, marker="o")
    plt.xlabel("Model size parameter")
    plt.ylabel("Rényi-2 entropy")
    plt.title(title)
    plt.tight_layout()
    plt.show()

    # vs actual param count
    plt.figure(figsize=(8, 5))
    plt.plot(n_params, S2, marker="o")
    plt.xlabel("Number of parameters")
    plt.ylabel("Rényi-2 entropy")
    plt.title(title + " (vs parameter count)")
    plt.tight_layout()
    plt.show()


def plot_seed_averaging(avg_result, title="Disorder averaging"):
    mean = float(avg_result["S2_mean"])
    std = float(avg_result["S2_std"])

    plt.figure(figsize=(6, 4))
    plt.bar(["S2"], [mean], yerr=[std], capsize=5)
    plt.ylabel("Rényi-2 entropy")
    plt.title(title + f"\n(mean ± std = {mean:.3f} ± {std:.3f})")
    plt.tight_layout()
    plt.show()

def plot_architecture_entropy_scaling(results, title=None, show_errorbars=True):
    """
    Plot disorder-averaged Renyi-2 entropy vs subsystem size
    for RBM, FFN, CNN.
    """
    plt.figure(figsize=(8, 5))

    for model_name, res in results.items():
        x = res["subsystem_sizes"]
        y = res["mean"]
        yerr = res["std"]

        if show_errorbars:
            plt.errorbar(x, y, yerr=yerr, marker="o", capsize=4, label=model_name)
        else:
            plt.plot(x, y, marker="o", label=model_name)

    plt.xlabel("Subsystem size")
    plt.ylabel(r"Rényi-2 entropy $S_2$")
    plt.title(title if title is not None else "Rényi-2 entropy scaling with subsystem size")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_swap_vs_exact_entropy(result, title=None, show_errorbars=True):
    x = result["subsystem_sizes"]

    plt.figure(figsize=(8, 5))

    if show_errorbars:
        plt.errorbar(
            x,
            result["swap_mean"],
            yerr=result["swap_std"],
            marker="o",
            capsize=4,
            label=r"Swap-estimated $S_2$",
        )
        plt.errorbar(
            x,
            result["exact_mean"],
            yerr=result["exact_std"],
            marker="s",
            capsize=4,
            label=r"Exact $S_2$",
        )
    else:
        plt.plot(x, result["swap_mean"], marker="o", label=r"Swap-estimated $S_2$")
        plt.plot(x, result["exact_mean"], marker="s", label=r"Exact $S_2$")

    plt.xlabel("Subsystem size")
    plt.ylabel(r"Rényi-2 entropy $S_2$")
    plt.title(title if title is not None else f"{result['model']}: swap vs exact")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_swap_vs_exact_multiple(results):
    for name, res in results.items():
        plot_swap_vs_exact_entropy(res, title=name)

def plot_entropy_vs_parameter_count(scan_results, title=None, show_errorbars=True):
    """
    Plot one architecture: disorder-averaged S2 vs number of parameters.
    """
    x = [r["n_parameters"] for r in scan_results]
    y = [float(r["S2_mean"]) for r in scan_results]
    yerr = [float(r["S2_std"]) for r in scan_results]

    plt.figure(figsize=(8, 5))

    if show_errorbars:
        plt.errorbar(x, y, yerr=yerr, marker="o", capsize=4)
    else:
        plt.plot(x, y, marker="o")

    plt.xlabel("Number of parameters")
    plt.ylabel(r"Rényi-2 entropy $S_2$")
    plt.title(title if title is not None else "Entropy vs parameter count")
    plt.tight_layout()
    plt.show()

### CNN LAYERS PLOT
from random_state_experiments import compare_swap_vs_exact_for_one_model

def plot_cnn_depths_swap_vs_exact(
    *,
    L=10,
    seeds=range(10),
    subsystem_sizes=None,
    cnn_channels=16,
    cnn_kernel=3,
    depths=(1, 2, 3, 4, 5),
    entropy_pairings=16,
    nchains=32,
    nsamples_per_chain=16,
    neq=50,
    nskip=2,
    show_errorbars=True,
):
    """
    Make 5 subfigures for CNN depths 1 to 5, each comparing:
      - swap-estimated Renyi-2 entropy
      - exact Renyi-2 entropy

    Only use for small L, since the exact entropy is computed from the
    full neural-network wavefunction by basis enumeration. 
    """
    if subsystem_sizes is None:
        subsystem_sizes = list(range(1, L // 2 + 1))

    fig, axes = plt.subplots(1, len(depths), figsize=(4 * len(depths), 4), sharey=True)

    if len(depths) == 1:
        axes = [axes]

    for ax, depth in zip(axes, depths):
        res = compare_swap_vs_exact_for_one_model(
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

        x = res["subsystem_sizes"]

        if show_errorbars:
            ax.errorbar(
                x,
                res["swap_mean"],
                yerr=res["swap_std"],
                marker="o",
                capsize=3,
                label=r"Swap $S_2$",
            )
            ax.errorbar(
                x,
                res["exact_mean"],
                yerr=res["exact_std"],
                marker="s",
                capsize=3,
                label=r"Exact $S_2$",
            )
        else:
            ax.plot(x, res["swap_mean"], marker="o", label=r"Swap $S_2$")
            ax.plot(x, res["exact_mean"], marker="s", label=r"Exact $S_2$")

        ax.set_title(f"{depth} layer" if depth == 1 else f"{depth} layers")
        ax.set_xlabel("Subsystem size")

    axes[0].set_ylabel(r"Rényi-2 entropy $S_2$")
    axes[0].legend()

    fig.suptitle(f"CNN: swap vs exact Rényi-2 entropy, L={L}")
    plt.tight_layout()
    plt.show()

from random_state_experiments import disorder_averaged_entropy_profile

def plot_cnn_entropy_vs_kernel(
    *,
    L=10,
    seeds=range(10),
    subsystem_sizes=None,
    kernel_sizes=(1, 3, 5, 7),
    cnn_channels=16,
    cnn_n_conv_layers=2,
    entropy_pairings=16,
    nchains=32,
    nsamples_per_chain=16,
    neq=50,
    nskip=2,
    show_errorbars=True,
    title=None,
):
    """
    Plot disorder-averaged Renyi-2 entropy vs subsystem size
    for different CNN kernel sizes.
    """
    if subsystem_sizes is None:
        subsystem_sizes = list(range(1, L // 2 + 1))

    plt.figure(figsize=(8, 5))

    for kernel in kernel_sizes:
        res = disorder_averaged_entropy_profile(
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
            kernel=kernel,
            n_conv_layers=cnn_n_conv_layers,
        )

        x = res["subsystem_sizes"]
        y = res["mean"]
        yerr = res["std"]

        if show_errorbars:
            plt.errorbar(
                x, y, yerr=yerr, marker="o", capsize=4, label=f"kernel={kernel}"
            )
        else:
            plt.plot(x, y, marker="o", label=f"kernel={kernel}")

    plt.xlabel("Subsystem size")
    plt.ylabel(r"Rényi-2 entropy $S_2$")
    plt.title(
        title
        if title is not None
        else f"CNN entropy scaling vs kernel size, L={L}"
    )
    plt.legend()
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
import jax.numpy as jnp

from random_state_experiments import disorder_averaged_entropy_profile


def plot_entropy_profiles_vs_subsystem_size_for_sizes(
    *,
    L=16,
    seeds=range(10),
    subsystem_sizes=None,
    entropy_pairings=16,
    nchains=32,
    nsamples_per_chain=16,
    neq=50,
    nskip=2,
):
    if subsystem_sizes is None:
        subsystem_sizes = list(range(1, L // 2 + 1))

    # -------------------------
    # RBM: small / medium / large
    # -------------------------
    rbm_configs = {
        "small": {"hidden": 4},
        "medium": {"hidden": 24},
        "large": {"hidden": 36},
    }

    plt.figure(figsize=(8, 5))
    for label, cfg in rbm_configs.items():
        res = disorder_averaged_entropy_profile(
            "RBM",
            L=L,
            seeds=seeds,
            subsystem_sizes=subsystem_sizes,
            entropy_pairings=entropy_pairings,
            nchains=nchains,
            nsamples_per_chain=nsamples_per_chain,
            neq=neq,
            nskip=nskip,
            hidden=cfg["hidden"],
        )

        x = jnp.array(res["subsystem_sizes"])
        y = jnp.array(res["mean"])
        yerr = jnp.array(res["std"])

        plt.errorbar(
            x, y, yerr=yerr,
            marker="o", capsize=4,
            label=f"{label} ({cfg['hidden']} hidden)"
        )

    plt.xlabel("Subsystem size")
    plt.ylabel(r"Rényi-2 entropy $S_2$")
    plt.title(f"RBM: entropy scaling vs subsystem size, L={L}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # -------------------------
    # FFN: small / medium / large
    # -------------------------
    ffn_configs = {
        "small": {"hidden_layers": (8,)},
        "medium": {"hidden_layers": (32, 32)},
        "large": {"hidden_layers": (64, 64)},
    }

    plt.figure(figsize=(8, 5))
    for label, cfg in ffn_configs.items():
        res = disorder_averaged_entropy_profile(
            "FFN",
            L=L,
            seeds=seeds,
            subsystem_sizes=subsystem_sizes,
            entropy_pairings=entropy_pairings,
            nchains=nchains,
            nsamples_per_chain=nsamples_per_chain,
            neq=neq,
            nskip=nskip,
            hidden_layers=cfg["hidden_layers"],
        )

        x = jnp.array(res["subsystem_sizes"])
        y = jnp.array(res["mean"])
        yerr = jnp.array(res["std"])

        plt.errorbar(
            x, y, yerr=yerr,
            marker="o", capsize=4,
            label=f"{label} {cfg['hidden_layers']}"
        )

    plt.xlabel("Subsystem size")
    plt.ylabel(r"Rényi-2 entropy $S_2$")
    plt.title(f"FFN: entropy scaling vs subsystem size, L={L}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # -------------------------
    # CNN: small / medium / large
    # -------------------------
    cnn_configs = {
    "small": {"channels": 4, "n_conv_layers": 1},
    "medium": {"channels": 16, "n_conv_layers": 2},
    "large": {"channels": 16, "n_conv_layers": 3},
    }

    plt.figure(figsize=(8, 5))
    for label, cfg in cnn_configs.items():
        res = disorder_averaged_entropy_profile(
            "CNN",
            L=L,
            seeds=seeds,
            subsystem_sizes=subsystem_sizes,
            entropy_pairings=entropy_pairings,
            nchains=nchains,
            nsamples_per_chain=nsamples_per_chain,
            neq=neq,
            nskip=nskip,
            channels=cfg["channels"],
            kernel=3,
            n_conv_layers=cfg["n_conv_layers"],
        )

        x = jnp.array(res["subsystem_sizes"])
        y = jnp.array(res["mean"])
        yerr = jnp.array(res["std"])

        plt.errorbar(
            x, y, yerr=yerr,
            marker="o", capsize=4,
            label=f"{label} (ch={cfg['channels']}, layers={cfg['n_conv_layers']})"
        )

    plt.xlabel("Subsystem size")
    plt.ylabel(r"Rényi-2 entropy $S_2$")
    plt.title(f"CNN: entropy scaling vs subsystem size, L={L}")
    plt.legend()
    plt.tight_layout()
    plt.show()