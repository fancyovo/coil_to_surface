def plot_poincare_from_dofs(
    surf_dofs,
    coil_dofs,
    currents,
    nfp,
    mpol=10,
    ntor=10,
    coil_order=16,
    coil_quadpoints=300,
    nfieldlines=20,
    radial_width=None,
    edge_fraction=1.1,
    stop_fraction=1.5,
    tmax_fl=2.0e8,
    phis=None,
    save_path=None,
    tol=1e-5,
    plot_surface=True,
    plot=True,
    marker_size=10,
):
    """Trace field lines from surface/coil DOFs and save a Poincare plot.

    Currents are normalized by their mean absolute value before tracing. This
    keeps relative current signs/ratios while making ``tmax_fl`` the tracing
    length control independent of the generated current norm.
    """
    surface = SurfaceXYZTensorFourier(
        mpol=int(mpol),
        ntor=int(ntor),
        nfp=int(nfp),
        stellsym=True,
    )
    surface.set_dofs(np.asarray(surf_dofs, dtype=float).ravel())

    coil_dofs = np.atleast_2d(np.asarray(coil_dofs, dtype=float))
    curves_half = []
    for row in coil_dofs:
        curve = CurveXYZFourier(int(coil_quadpoints), int(coil_order))
        curve.set_dofs(row.ravel())
        curves_half.append(curve)

    currents = np.atleast_1d(np.asarray(currents, dtype=float)).ravel()
    current_mean = float(np.mean(np.abs(currents)))
    trace_currents = currents / current_mean if current_mean > 0 else currents
    print(f"Poincare currents normalized by mean|I|={current_mean:.6g}")
    current_objs = [Current(float(val)) for val in trace_currents]
    for current in current_objs:
        current.fix_all()

    coils = coils_via_symmetries(curves_half, current_objs, int(nfp), True)
    bs = BiotSavart(coils)

    gamma0 = surface.gamma()[0]
    r0 = np.linalg.norm(gamma0[:, :2], axis=1)
    z0 = gamma0[:, 2]
    seed_r = float(0.5 * (np.min(r0) + np.max(r0)))
    seed_z = float(np.mean(z0))

    if phis is None:
        phis = [(i / 4) * (2 * np.pi / int(nfp)) for i in range(4)]
    else:
        phis = list(np.asarray(phis, dtype=float).ravel())

    r_outer_phi0 = float(np.max(np.linalg.norm(surface.gamma()[0, :, :2], axis=1)))
    gamma = surface.gamma()
    r_all = np.linalg.norm(gamma[:, :, :2], axis=2)
    z_all = gamma[:, :, 2]
    r_center = 0.5 * (float(np.min(r_all)) + float(np.max(r_all)))
    z_center = 0.5 * (float(np.min(z_all)) + float(np.max(z_all)))
    r_half_width = 0.5 * (float(np.max(r_all)) - float(np.min(r_all))) * float(stop_fraction)
    z_half_width = 0.5 * (float(np.max(z_all)) - float(np.min(z_all))) * float(stop_fraction)
    stopping_criteria = [
        MinRStoppingCriterion(max(r_center - r_half_width, 0.0)),
        MaxRStoppingCriterion(r_center + r_half_width),
        MinZStoppingCriterion(z_center - z_half_width),
        MaxZStoppingCriterion(z_center + z_half_width),
    ]
    print(
        f"Stopping box: R=[{max(r_center - r_half_width, 0.0):.6g}, {r_center + r_half_width:.6g}], "
        f"Z=[{z_center - z_half_width:.6g}, {z_center + z_half_width:.6g}]"
    )
    r_stop = float(seed_r + float(edge_fraction) * (r_outer_phi0 - seed_r))
    if radial_width is None:
        r_end = r_stop
    else:
        r_end = min(seed_r + float(radial_width), r_stop)
    if r_end <= seed_r:
        r_end = r_stop
    R0 = np.linspace(seed_r, r_end, int(nfieldlines))
    Z0 = np.full(int(nfieldlines), seed_z)
    print(f"R0 range = [{R0[0]:.6g}, {R0[-1]:.6g}], boundary R(phi=0) ≈ {r_outer_phi0:.6g}")

    print("Beginning field line tracing")
    fieldlines_phi_hits = []
    for r_start, z_start in tqdm(
        list(zip(R0, Z0)),
        desc="Tracing fieldlines",
        unit="line",
    ):
        tys_one, hits_one = compute_fieldlines(
            bs,
            [r_start],
            [z_start],
            tmax=float(tmax_fl),
            tol=float(tol),
            phis=phis,
            stopping_criteria=stopping_criteria,
        )
        fieldlines_phi_hits.append(hits_one[0])
    hit_counts = [0 if np.asarray(hits).ndim != 2 else int(np.sum(np.asarray(hits)[:, 1] >= 0))
                  for hits in fieldlines_phi_hits]
    print("Poincare hits per line =", hit_counts)

    surf_arg = surface if plot_surface else None
    nrowcol = int(np.ceil(np.sqrt(len(phis))))
    fig, axs = plt.subplots(nrowcol, nrowcol, figsize=(8, 5))
    axs = np.asarray(axs).reshape((nrowcol, nrowcol))
    for i, phi in enumerate(phis):
        ax = axs[i // nrowcol, i % nrowcol]
        ax.set_aspect("equal")
        ax.set_title(f"$\\phi = {phi / np.pi:.2f}\\pi$ ", loc="left", y=0.0)
        if i // nrowcol == nrowcol - 1:
            ax.set_xlabel("$r$")
        if i % nrowcol == 0:
            ax.set_ylabel("$z$")
        ax.grid(True, linewidth=0.5)
        for hits in fieldlines_phi_hits:
            hits = np.asarray(hits)
            if hits.ndim != 2 or hits.shape[0] == 0:
                continue
            data = hits[np.where(hits[:, 1] == i)[0], :]
            if data.size == 0:
                continue
            r = np.sqrt(data[:, 2] ** 2 + data[:, 3] ** 2)
            ax.scatter(r, data[:, 4], s=marker_size, linewidths=0)
        if surf_arg is not None:
            # `compute_fieldlines` uses radians; `Surface.cross_section` expects phi/(2*pi).
            cross_section = surf_arg.cross_section(phi=phi / (2 * np.pi))
            r_interp = np.sqrt(cross_section[:, 0] ** 2 + cross_section[:, 1] ** 2)
            z_interp = cross_section[:, 2]
            ax.plot(r_interp, z_interp, linewidth=1, c="k")
    for i in range(len(phis), nrowcol * nrowcol):
        axs[i // nrowcol, i % nrowcol].axis("off")
    plt.tight_layout()
    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"Saved Poincare plot to {save_path}")
    if plot:
        plt.show()
    else:
        plt.close(fig)
    return fig
