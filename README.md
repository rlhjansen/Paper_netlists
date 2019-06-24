
# Replication:
## creating your own data:

```
python main.py gen x *circuit-width* y *circuit-depth*
```

leaving out x and y parameters will results in generated circuits used in the paper.

## running the experiment:

```
python main.py optimize tag *your tag*
```

runs are given tags to avoid overwriting files through git when collecting data on different machines

# directory structure:

1. code
    - algorithms
    - classes
    - visualization
2. data
    - baseline
    - generated
3. results
    - circuit size
        - generated series
            - circuits ordered by total gates
                - specific circuits
                    - netlists ordered by total nets
                        - specific netlist

# authors:

Reitze Jansen

# supervisor:

Daan van den Berg
