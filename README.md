# Hybrid Metaheuristic Neural Network Optimization

Benchmarking three nature-inspired hybrid algorithms for neural network 
weight optimization on a real-world mining regression dataset.

## Overview
Implements and compares three hybrid metaheuristic-neural network models 
to solve a complex regression task, moving beyond standard gradient descent 
to explore how nature-inspired search strategies optimize neural network parameters.

## Models Implemented
- HHO-MLP → Harris Hawks Optimization + Multi-Layer Perceptron
- SHO-ANN → Sandpiper Herd Optimization + Artificial Neural Network
- GSO-MLP → Glowworm Swarm Optimization + Multi-Layer Perceptron

## Results
| Model   | R²     | RMSE (norm) | MAE (norm) |
|---------|--------|-------------|------------|
| HHO-MLP | 0.9938 | 0.0316      | 0.0264     |
| SHO-ANN | 0.9989 | 0.0077      | 0.0059     |
| PSO-MLP | 0.9997 | 0.0618      | 0.0445     |

All models achieved R² ≈ 1.0 confirming strong regression performance.

## Tech Stack
Python | NumPy | Scikit-learn | Matplotlib


