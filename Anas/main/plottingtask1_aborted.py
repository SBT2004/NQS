import matplotlib.pyplot as plt


# ============================================================
# Internal helper: compute exact TFIM entropy data from ED
# ============================================================

def _prepare_tfim_entropy_data(
    *,
    L,
    J,
    g_values,
    subsystem_sizes=None,
):
    from ED import (
        exact_tfim_ground_state,
        von_neumann_entropy_from_statevector,
        renyi2_entropy_from_statevector,
    )

    if subsystem_sizes is None:
        subsystem_sizes = list(range(1, L // 2 + 1))

    data = {}

    for g in g_values:
        _, psi0 = exact_tfim_ground_state(L, J, g)

        SvN_vals = []
        S2_exact_vals = []

        for LA in subsystem_sizes:
            SvN_vals.append(
                float(von_neumann_entropy_from_statevector(psi0, L, LA))
            )
            S2_exact_vals.append(
                float(renyi2_entropy_from_statevector(psi0, L, LA))
            )

        data[g] = {
            "subsystem_sizes": list(subsystem_sizes),
            "SvN": SvN_vals,
            "S2_exact": S2_exact_vals,
        }

    return data


# ============================================================
# Plot: exact von Neumann entropy
# ============================================================

def plot_tfim_von_neumann(
    *,
    L=16,
    J=1.0,
    g_values=(1.0, 0.5),
    subsystem_sizes=None,
    title="Entanglement scaling in 1D TFIM",
    add_inset=True,
    inset_g=1.0,
    inset_range=(7, 8),
):
    data = _prepare_tfim_entropy_data(
        L=L,
        J=J,
        g_values=g_values,
        subsystem_sizes=subsystem_sizes,
    )

    plt.figure(figsize=(8, 5))

    for g in g_values:
        label = f"g={g} (critical)" if g == 1.0 else f"g={g} (non-critical)"
        plt.plot(
            data[g]["subsystem_sizes"],
            data[g]["SvN"],
            marker="o" if g == 1.0 else "s",
            label=label,
        )

    plt.xlabel(r"Subsystem size $L_A$")
    plt.ylabel(r"Von Neumann entropy $S_1$")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    if add_inset and inset_g in data:
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        x = data[inset_g]["subsystem_sizes"]
        y = data[inset_g]["SvN"]

        xmin, xmax = inset_range
        x_zoom = [xi for xi in x if xmin <= xi <= xmax]
        y_zoom = [yi for xi, yi in zip(x, y) if xmin <= xi <= xmax]

        if len(x_zoom) >= 2:
            ax = plt.gca()
            axins = inset_axes(ax, width="30%", height="35%", loc="upper right")
            axins.plot(x_zoom, y_zoom, marker="o")
            axins.set_xlim(min(x_zoom), max(x_zoom))

            pad = 0.1 * (max(y_zoom) - min(y_zoom) + 1e-8)
            axins.set_ylim(min(y_zoom) - pad, max(y_zoom) + pad)

            ax.indicate_inset_zoom(axins, edgecolor="black")

    plt.show()


# ============================================================
# Plot: exact Rényi-2 entropy
# ============================================================

def plot_tfim_renyi2(
    *,
    L=16,
    J=1.0,
    g_values=(1.0, 0.5),
    subsystem_sizes=None,
    title="Entanglement scaling in 1D TFIM",
    add_inset=True,
    inset_g=1.0,
    inset_range=(7, 8),
):
    data = _prepare_tfim_entropy_data(
        L=L,
        J=J,
        g_values=g_values,
        subsystem_sizes=subsystem_sizes,
    )

    plt.figure(figsize=(8, 5))

    for g in g_values:
        label = f"g={g} (critical)" if g == 1.0 else f"g={g} (non-critical)"
        plt.plot(
            data[g]["subsystem_sizes"],
            data[g]["S2_exact"],
            marker="o" if g == 1.0 else "s",
            label=label,
        )

    plt.xlabel(r"Subsystem size $L_A$")
    plt.ylabel(r"Rényi-2 entropy $S_2$")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    if add_inset and inset_g in data:
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        x = data[inset_g]["subsystem_sizes"]
        y = data[inset_g]["S2_exact"]

        xmin, xmax = inset_range
        x_zoom = [xi for xi in x if xmin <= xi <= xmax]
        y_zoom = [yi for xi, yi in zip(x, y) if xmin <= xi <= xmax]

        if len(x_zoom) >= 2:
            ax = plt.gca()
            axins = inset_axes(ax, width="30%", height="35%", loc="upper right")
            axins.plot(x_zoom, y_zoom, marker="o")
            axins.set_xlim(min(x_zoom), max(x_zoom))

            pad = 0.1 * (max(y_zoom) - min(y_zoom) + 1e-8)
            axins.set_ylim(min(y_zoom) - pad, max(y_zoom) + pad)

            ax.indicate_inset_zoom(axins, edgecolor="black")

    plt.show()
