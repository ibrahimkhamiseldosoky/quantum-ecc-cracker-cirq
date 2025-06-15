
# 🔐 Quantum ECC Toy Attack – Cirq Implementation

This project demonstrates a functional quantum attack on a toy elliptic curve cryptosystem using Google's [Cirq](https://github.com/quantumlib/Cirq). It targets the Elliptic Curve Discrete Logarithm Problem (ECDLP) using a custom-built quantum oracle combined with Grover's algorithm, and successfully recovers private keys in small prime fields.

---

## 📌 Submission Summary

- **Name**: Ibrahim Khamis
- **Email**: ibrahimkhamiseldosoky@gmail.com
- **Cracked Key Size**: 3 to 5 bits
- **Platform**: Cirq (v1.5.0) simulator on local CPU
- **Hardware**: Windows 11, 28-core Intel Xeon, 32GB RAM (no GPU)
- **Success Rate**: 9.09% over 11 toy ECC experiments
- **Average Quantum Runtime**: 0.0423s per circuit

---

## 🧠 Project Highlights

- Fully custom ECC implementation for small fields (mod 5, 7, 11, 13)
- Toy private keys like `2`, `3`, `5`, `7`, etc.
- Cirq-based quantum oracle design with Grover-style amplification
- Classical baseline used to verify correctness
- Output includes performance stats, probability distributions, and circuit depth analysis
- Graphical result visualizations included (auto-generated)

---

## 🛠️ How to Run

### 1. Requirements
Install the following with pip:
```bash
pip install cirq numpy matplotlib psutil
````

### 2. Run the Simulation

```bash
python main.py
```

### 3. Output

* Success/failure per curve
* Quantum vs classical timings
* Circuit resource usage
* Plots (auto-shown)
* Full analysis saved to `quantum_ecc_cirq_results.json`

---

## 📄 Included Files

* `python.py` – Main ECC simulator + quantum attacker
* `brief.pdf` – Summary document (2 pages)
* `quantum_ecc_cirq_results.json` – Results and metadata
* `README.md` – This file

---

## 📬 Submission for Q-Day Prize

This repository is part of a submission for the [Project 11 Q‑Day Prize](https://project-eleven.dev/qdayprize), demonstrating a successful quantum discrete logarithm attack on elliptic curve cryptography in toy settings.

---

*Built and tested by Ibrahim Khamis, 13-year-old researcher from Egypt.*


