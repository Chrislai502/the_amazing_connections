# Making Connections

This repository contains tools and solvers for playing and evaluating **Connections** games with AI models. Our solvers implement various strategies—ranging from basic approaches to more advanced multi-agent, dual-process frameworks designed to mitigate “analysis paralysis.”

## **Background**

When Large Language Models (LLMs) tackle iterative puzzles like *Connections*, they can sometimes get stuck in repetitive loops (overthinking or “analysis paralysis”). Inspired by **Rational Speech Act** (RSA) theory and **dual-process** cognition (System 1 vs. System 2), we’ve developed:
- **GVC (Guess, Validate, Consensus)**: A multi-agent system that separates guessing from validation, ensuring more grounded proposals.  
- **Snap GVC**: An enhanced version of GVC that quickly switches between slow, deliberative reasoning (System 2) and fast, intuitive guesses (System 1) when stagnation is detected—mitigating overthinking and improving puzzle-solving efficiency.

> **Note:** In the paper, these approaches are referred to as Think, Validate, Consensus (TVC) and Snap-Think. The code here uses the analogous terms GVC and Snap GVC.

---

## **Features**

- **Multiple solvers**:
  - **Naive / Basic**: Simple heuristics without chain-of-thought.
  - **CoT**: Chain-of-Thought prompting.
  - **GVC (Guess, Validate, Consensus)**: Multi-agent approach to reduce incorrect or ungrounded guesses.
  - **Snap GVC**: Dual-process version that switches to quick, high-temperature guesses upon detecting repeated failures.
- **Flexible model support**: Works with **GPT-4o**, **Llama-3.3**, etc.
- **Evaluation & Benchmarks**: Automated scripts to measure solver performance across multiple games.

---

## **Installation**

Create a conda environment and activate it:

```bash
conda create -n connections python=3.12 -y
conda activate connections
```

Clone the repository and install dependencies:

```bash
git clone https://github.com/Chrislai502/the_amazing_connections.git
cd the_amazing_connections
pip install -e .
```

---

## **Running a Demo**

The main script is `run.py`. Below are some example commands.

### **Recommended Demo: Snap GVC + GPT-4o**

```bash
python src/rsallms/run.py snap_gvc gpt-4o --start 0 --end 10
```

- **Snap GVC**: Uses a dual-process approach (similar to System 1 vs. System 2) to avoid analysis paralysis.  
- **GPT-4o**: Recommended for robust language reasoning.

### **General Usage**

```bash
python src/rsallms/run.py <solver_type> <model> --start <start_index> --end <end_index>
```

- `<solver_type>`: `naive`, `cot`, `basic`, `gvc`, or `snap_gvc`
- `<model>`: e.g. `llama-3.3-70b-versatile`, `llama-3.1-8b-instant`, `gpt-4o`, `gpt-4o-mini`
- `--start`, `--end`: Specify the range of puzzle indices.

**Example**:

```bash
python src/rsallms/run.py cot llama-3.3-70b-versatile --start 5 --end 20
```
Runs the CoT solver with LLaMA-3.3-70b on puzzles [5..20].

---

## **Switching Models**

Just change the `<model>` argument:

- **GPT-4o**:

  ```bash
  python src/rsallms/run.py snap_gvc gpt-4o --start 0 --end 5
  ```

- **LLaMA-3.3-70b**:

  ```bash
  python src/rsallms/run.py gvc llama-3.3-70b-versatile --start 10 --end 20
  ```

**GPT-4o** is recommended for the best results with Snap GVC.

---

## **Adding Games**

The script loads puzzles via `load_games()`. To add or edit puzzles:

1. Update the relevant Connections data files.  
2. Ensure each game follows the expected format for the solver classes.

---

## **Paper & Reference**

For a deeper look at the multi-agent, dual-process approach used here, see our paper:

> **Title**: *Snap Out of It: A Dual-Process, Multi-Agent Framework to Mitigate Analysis Paralysis in LLMs*  
> **Authors**: [Ashish Pandian](mailto:ashishpandian@berkeley.edu), [Chris Lai](mailto:chris.lai@berkeley.edu), [Nelson Lojo](mailto:nelson.lojo@berkeley.edu), [Jackson Lukas](mailto:jacksonlukas@berkeley.edu)

In the code, we use the terms **GVC** and **Snap GVC** to describe the same ideas (TVC & Snap-Think) from the paper.

---

## **Maintainer**

This repository is maintained by:

**Chris Lai**  
Email: [chrislai_502@berkeley.edu](mailto:chrislai_502@berkeley.edu)
