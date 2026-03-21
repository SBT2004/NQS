import matplotlib.pyplot as plt


def plot_training_curves(result, title_prefix=None):
    model = result["model"]
    exact_energy = result["exact_energy"]
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
    plt.plot(g_vals, S_vals, marker="o")
    plt.title(title + " - Rényi-2 entropy")
    plt.xlabel("g")
    plt.ylabel("S2")
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