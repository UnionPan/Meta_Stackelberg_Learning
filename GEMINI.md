# GEMINI - Meta Reinforcement Learning for Stackelberg Games in Federated Learning

## Project Overview

This project simulates a security scenario in Federated Learning (FL) using a Meta-Reinforcement Learning (Meta-RL) framework. The core of the simulation is a Stackelberg game where a defender and an attacker compete. The defender, acting as the leader, learns to anticipate and counter data poisoning and backdoor attacks by dynamically adjusting the hyperparameters of aggregation rules. The attacker, as the follower, adapts its strategy to the defender's actions.

## Problem Formulation: Stackelberg Game

The interaction between the defender and attacker is modeled as a Stackelberg game:

-   **Leader (Defender):** The central server in the FL system. The defender's goal is to learn a meta-policy that selects the optimal hyperparameters for the aggregation rule to maintain model accuracy and robustness.
-   **Follower (Attacker):** A subset of malicious clients in the FL system. The attacker's goal is to degrade the global model's performance or create backdoors by sending malicious updates.

### State Space

The state `s_t` at each round `t` includes:

-   The current global model parameters.
-   The history of local updates.
-   The previous actions of the defender and attacker.
-   Statistical properties of the submitted model updates.

### Action Space

-   **Defender's Actions:** A set of hyperparameters for various aggregation rules (e.g., `n-rum`, `trimmed-mean`, `krum`). This could include the number of clients to trim, the trimming percentage, etc.
-   **Attacker's Actions:** The type and intensity of the attack (e.g., percentage of poisoned data, target label for a backdoor).

### Reward Structure

-   **Defender's Reward:** A function of the global model's accuracy on a held-out test set, and potentially a penalty for the computational cost of the defense.
-   **Attacker's Reward:** A function of the global model's misclassification rate on the target class (for backdoor attacks) or overall performance degradation.

## Environment: Federated Learning

-   **FL Setup:** A standard federated learning environment with a central server and a number of clients. A fraction of the clients are malicious.
-   **Dataset:** Standard image classification datasets like CIFAR-10 or MNIST will be used initially.
-   **Models:** Convolutional Neural Networks (CNNs) suitable for the chosen datasets.

## Adversary Model

The attackers employ a mixed strategy of:

-   **Data Poisoning:** Malicious clients poison their local training data to skew the global model.
-   **Backdoor Attacks:** Malicious clients insert a backdoor into the global model by training on data with a specific trigger.

## Defense Mechanism

The defender uses a variety of aggregation rules to protect the global model. The meta-RL agent will learn to select the best rule and its hyperparameters at each communication round. Examples of aggregation rules include:

-   **FedAvg:** Standard federated averaging.
-   **Krum / Multi-Krum:** Selects the model update that is closest to its neighbors.
-   **Trimmed Mean:** Trims a certain percentage of the highest and lowest model updates.
-   **Coordinate-wise Median:** Computes the median of the model updates for each coordinate.

## Meta-Reinforcement Learning Approach

A meta-RL algorithm (like MAML or REPTILE) will be used to train the defender. The goal is to learn a policy that can quickly adapt to new attack strategies. The meta-learning process involves:

1.  **Inner Loop:** The defender (agent) interacts with the attacker over a series of FL rounds, learning a policy for a specific, fixed attack strategy.
2.  **Outer Loop:** The meta-policy is updated based on the performance of the policies learned in the inner loop across a distribution of different attack strategies.

This allows the defender to generalize and effectively counter a wide range of attacks, even novel ones.

## Simulation Setup

-   **Libraries:**
    -   **PyTorch** or **TensorFlow** for building and training the models.
    -   **PySyft** (optional) for a more realistic FL simulation.
    -   **NumPy** for numerical operations.
    -   **Gymnasium (formerly OpenAI Gym)** for structuring the RL environment.

-   **Simulation Flow:**
    1.  Initialize the FL environment, clients, and the global model.
    2.  The meta-RL agent (defender) selects an aggregation rule and its hyperparameters.
    3.  Clients train the model on their local data (malicious clients poison their data).
    4.  Clients send their model updates to the server.
    5.  The server applies the chosen aggregation rule to the updates.
    6.  The global model is updated.
    7.  The defender receives a reward based on the model's performance.
    8.  The meta-RL agent's policy is updated.
    9.  Repeat for a set number of communication rounds.

## Evaluation Metrics

-   **Global Model Accuracy:** The accuracy of the final global model on a clean test set.
-   **Attack Success Rate:** The success rate of the backdoor attack on the target class.
-   **Robustness:** The performance of the defense against a variety of attack strategies.
-   **Convergence Speed:** The number of communication rounds required for the global model to converge.

## Expected Outcomes

-   A robust meta-RL agent that can effectively defend a Federated Learning system against a range of data poisoning and backdoor attacks.
-   Insights into the dynamics of the Stackelberg game between attackers and defenders in FL.
-   A flexible and extensible simulation framework for future research in FL security.

## Development Progress (as of 2025-09-09)

### 1. Project Scaffolding

- The initial directory structure has been created to organize the project's code, data, and results.

### 2. Code Extraction and Refactoring

- The core logic from the experimental notebook (`notebooks/Mixed_RLdefendFL4.ipynb`) has been extracted and modularized into the `src` directory.
- **Models (`src/models/cnn.py`):** ResNet and VGG model architectures have been saved.
- **Data Utilities (`src/utils/data_loader.py`):** Functions for data loading, backdoor pattern generation, and dataset poisoning have been organized.
- **FL Utilities (`src/utils/fl_utils.py`):** A comprehensive set of functions for the FL simulation has been created, including:
    - Basic utilities for model weight manipulation.
    - Various aggregation rules (e.g., FedAvg, Krum, Median) that will serve as the defender's action space.
    - Functions for crafting adversarial attacks.

### 3. Code Quality and Verification

- **Commenting:** The extracted code has been commented to improve readability and maintainability.
- **Testing:** Basic unit tests have been created in the `tests` directory to ensure the correctness of the data loaders and core FL utilities.

### Next Steps

- Integrate the user-provided Reptile, environment, and agent definitions.
- Implement the `fl_env.py` to create the Gym-like environment for the RL agent.
- Implement the defender and attacker agents in `src/agents/`.
- Begin the implementation and training of the meta-RL agent.
