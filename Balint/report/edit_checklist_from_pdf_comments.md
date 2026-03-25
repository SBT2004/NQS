# Edit Checklist From PDF Comments

Source annotations were extracted from:

- `exercise_1_with_comments_backup.pdf`
- `exercise_2_with_comments_backup.pdf`
- `exercise_3.pdf`

The rewrite-only and research-only items have been applied in the current `.tex` sources. The remaining tasks are the ones that still require new runs, regenerated figures, or code-backed validation.

## Requires Running Code

### Exercise 2

- [ ] Double-check that the marked formula is correct against the implementation and generated outputs.

### Exercise 3

- [ ] Increase the number of steps in the relevant run/figure setup.
- [ ] Use periodic boundary conditions where indicated.
- [ ] Run the expanded three-system study:
- [ ] 1D TFIM chain of length 32 with PBC.
- [ ] 2D TFIM on a `5x5` lattice.
- [ ] J1-J2 on a `5x5` lattice, using a frustration ratio near `J2/J1 \approx 0.5` unless the final literature pass suggests a better benchmark point.
- [ ] Present results for all three architectures on those systems.
- [ ] For TFIM, show scaling behavior for `h = 0.5`, `1`, and `1.5`.
- [ ] Regenerate the relevant figures, tables, and the Exercise 3 discussion/conclusion once the new runs are available.
