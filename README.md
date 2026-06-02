# Repository for "Improved Lower Bounds for Proportionally Fair Centroid Clustering"

## Verification

The `symbolic-verification.ipynb` notebook symbolically verifies that the tight examples in **Theorem 5.7** to **5.9** form valid metric spaces. In particular, it checks the relevant 3-path triangle inequalities symbolically.
Furthermore, it also checks symbolically that calculations in the proofs of **Theorem 5.8** and **Lemma C.1** to **C.6** are valid.

## 37-point metric instance certificate

The `lowerBound_certificate` folder contains the generated numerical certificate for the 37-point instance reported in **Appendix C**.

- `distance_matrix37.csv`: pairwise distances between centers `c0,...,c36` and agents `a0,...,a36`.
- `weights37.csv`: feasible weight vector at `alpha = 2.15082963`.
- `solver_log_metric37.txt`: solver output and binary-search log.

The verifier and input instance used to generate/check these files are:

- `verify_instance.py`: runs the LP feasibility check, binary search over `alpha`, and the weight feasibility check.
- `instance_io.py`: small CSV loader for set-system instances.
- `instances/metric37_wide.csv`: CSV encoding of the deviation sets in the 37-point instance.

The verifier requires Gurobi and `gurobipy`.

### To verify the default 37-point instance, run:

```bash
python verify_instance.py
```

To save the solver output to a log file, run:

```bash
python verify_instance.py --log-file lowerBound_certificate/solver_log_37.txt
```

### To test your own instances:

```bash
python verify_instance.py --instance instances/your_instance.csv
```

The input CSV may use either long format:

```csv
agent,set
0,1
0,2
1,2
1,3
1,6
```

or wide format:

```csv
agent,set
0,"1 2"
1,"2,3;6"
```

In both formats, a row means that agent `j` belongs to set `S_t`. The wide `set` column accepts spaces, commas, or semicolons as separators.
