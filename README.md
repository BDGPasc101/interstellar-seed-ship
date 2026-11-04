# Relativistic Seed Ship Simulator 🚀

A high-fidelity systems-engineering and probabilistic risk assessment tool for modeling interstellar transit at relativistic speeds.

This simulator transcends standard kinematic calculators by implementing **coupled engineering feedback loops**, **distance-based material ablation**, and a **memory-optimized Monte Carlo hazard engine**. It evaluates the viability of structural shield geometries and propulsion systems against the harsh reality of the Interstellar Medium (ISM).

## 🌌 Project Overview

Traveling at significant fractions of the speed of light ($v/c \ge 0.10$) introduces extreme engineering challenges. Microscopic dust grains strike with the kinetic energy of tactical explosives, and the fuel required to reach these speeds forces ship designs into an exponential mass penalty. 

This interactive Streamlit dashboard allows users to design a needle-profile shield and propulsion architecture, run stochastic survival simulations, and find the mathematical convergence point where payload, fuel, and shielding mass balance.

## 🔬 Core Physics & Engineering Engines

### 1. Dual-Mode Structural Degradation
The interstellar medium follows an MRN power-law dust distribution ($n(r) \propto r^{-3.5}$). The simulator evaluates hull integrity across two distinct regimes:
*   **Micro-Ablation (Deterministic):** Trillions of sub-micron grains continuously sandblast the hull. This is modeled analytically as a distance-based volumetric erosion rate, dynamically thinning the armor over the flight path.
*   **Macroscopic Puncture (Stochastic):** Rare, large dust grains cause discrete, massive cratering. Modeled via a memory-optimized Monte Carlo engine.

### 2. Coupled Mass-Area Feedback Loop (The Death Spiral)
Spacecraft design is a closed-loop problem. This project implements an iterative numerical solver that couples the **Relativistic Rocket Equation** to the shield geometry:
*   Adding payload mass requires more fuel.
*   More fuel requires wider fuel tanks.
*   Wider tanks increase the ship's frontal area, requiring a wider, heavier shield.
*   A heavier shield requires even more fuel.

The solver iterates until the structural mass converges or diverges to infinity (highlighting the strict velocity limits of low-efficiency engines like Nuclear Fusion).

### 3. Vectorized Monte Carlo Simulation
Simulating interstellar transit involves evaluating trillions of potential particle interactions. To prevent memory overflow and CPU bottlenecking:
*   Utilizes continuous degradation to bypass simulating trillions of sub-micron grains.
*   Applies a Poisson/Gaussian hybrid generation model to sample macroscopic hazards dynamically.
*   Fully vectorized in NumPy to run 10,000 independent flight passes across light-years of space in seconds.

### 4. Optimal Trajectory Solver
Integrates `scipy.optimize.minimize_scalar` to balance transit time (time dilation) against the exponentially rising risk of structural failure, finding the exact optimal cruise velocity ($\beta^*$) for a given target system and shield architecture.

## 🛠️ Tech Stack & Architecture

*   **Frontend:** Streamlit, Plotly Express, Plotly Graph Objects
*   **Math & Physics Computation:** NumPy (Vectorization), SciPy (Optimization & Convergence Solving)
*   **Data Handling:** Pandas
*   **Architecture:** Modular design separating UI (`app.py`) from core physics engines (`relativistic.py`, `impacts.py`).

## 🚀 Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/interstellar-seed-ship.git
   cd interstellar-seed-ship
   ```

2. **Set up a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the simulator:**
   ```bash
   streamlit run app.py
   ```

## 📂 Repository Structure

*   `app.py`: Main Streamlit dashboard and UI routing.
*   `physics/relativistic.py`: Trajectory math, Lorentz transformations, rocket equation loops, and SciPy optimizers.
*   `physics/impacts.py`: Material science, MRN dust distributions, oblique relativistic cratering, and the Monte Carlo engine.
*   `data/seed_ship_targets.csv`: Catalog of exoplanet targets and their respective ISM density multipliers.
*   `learning.md`: Detailed documentation on the physics formulas, mathematical proofs, and architectural design decisions.