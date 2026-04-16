# Fractional Wave Equation with Caputo Memory (SOE)

This repository contains the Python source code used for the numerical experiments and simulations presented in our manuscript on the numerical methods for the fractional wave equation with Caputo memory. 

## Overview
The code implements and compares three different spatial discretization methods combined with the Newmark time-integration scheme and the Sum of Exponentials (SOE) fast memory evaluation algorithm:
1. **Finite Difference Method (FDM)**
2. **Finite Element Method (FEM)**
3. **Spectral Chebyshev Method (CGL)**

All methods use a unified Chebyshev-Gauss-Lobatto (CGL) grid to eliminate spatial bias during comparison. 

## File Structure
- `experiments.py` — Core numerical solvers (FDM, FEM, Spectral) and matrix assembly.
- `exp_solution.py` — Script to generate and plot the exact vs numerical solution profiles (wave propagation).
- `exp_parallel.py` — Benchmarks for parallel execution and performance scaling (Speedup evaluation).
- `exp_application.py` — Scripts for practical application scenarios (e.g., Seismic waves).
- `test_soe.py`, `test_temporal*.py`, `test_fm.py` — Unit tests and convergence validation scripts for the SOE algorithm and temporal integrators.
- `requirements.txt` — Required Python packages (NumPy, SciPy, Matplotlib).

## Installation
To run the code, ensure you have Python 3.8+ installed. Install the dependencies via pip:
```bash
pip install -r requirements.txt
```

## Usage
Simply run any of the `exp_*.py` scripts to reproduce the figures and tables presented in the manuscript. For example, to generate the wave profile evolution plot:
```bash
python exp_solution.py
```
