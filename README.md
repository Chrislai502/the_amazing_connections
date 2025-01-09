# Making Connections

This repository contains tools and solvers for playing and evaluating games of **Connections** with AI models. The solvers implement various strategies, including CoT (Chain-of-Thought), GVC (Guess, Validate, Consensus), and an advanced version called Snap GVC, which is optimized for large models like GPT-4o.

---

## **Features**

-   Multiple solvers: Naive, CoT, Basic, GVC, and Snap GVC.
-   Flexible support for AI models, including **GPT-4o**, **Llama-3.3**, and others.
-   Game evaluation and automated benchmarking.

---

## **Installation**
First, create a conda virtual environment and activate it:

```bash
conda create -n connections python=3.12 -y
conda activate avhubert
```

Clone the repository and install dependencies:

```bash
git clone https://github.com/Chrislai502/the_amazing_connections.git
cd the_amazing_connections
pip install -e .
```

---

## **Running a Demo**

The main demo script is `run.py`. Below are the steps to run it:

### **Quick Demo with Snap GVC and GPT-4o**

We recommend using the **Snap GVC solver** with the **GPT-4o** model for the best performance.

Run the following command:

```bash
python src/rsallms/run.py snap_gvc gpt-4o --start 0 --end 10
```

This runs the Snap GVC solver with the GPT-4o model on games indexed from 0 to 10 in the dataset.

---

### **General Usage**

To test with different solvers and models, use the following command format:

```bash
python src/rsallms/run.py <solver_type> <model> --start <start_index> --end <end_index>
```

#### **Arguments:**

-   `<solver_type>`: Choose from `naive`, `cot`, `basic`, `gvc`, or `snap_gvc`.
-   `<model>`: Supported models include:
    -   `llama-3.3-70b-versatile`
    -   `llama-3.1-8b-instant`
    -   `gpt-4o`
    -   `gpt-4o-mini`
-   `--start`: The starting index of the games to evaluate (default: 0).
-   `--end`: The ending index of the games to evaluate.

#### **Example:**

```bash
python src/rsallms/run.py cot llama-3.3-70b-versatile --start 5 --end 20
```

This runs the CoT solver using the LLaMA-3.3-70b-versatile model on games indexed from 5 to 20.

---

## **Switching Models**

To switch between models, simply provide the desired model name as an argument to `run.py`. For example:

-   Use **GPT-4o**:

    ```bash
    python src/rsallms/run.py snap_gvc gpt-4o --start 0 --end 5
    ```

-   Use **LLaMA-3.3-70b**:
    ```bash
    python src/rsallms/run.py gvc llama-3.3-70b-versatile --start 10 --end 20
    ```

For Snap GVC, **GPT-4o** is the top recommended model for optimal results.

---

## **Adding Games**

The script uses a `load_games()` function to load the dataset of games. To add or modify the games:

1. Update the data in the appropriate `Connections` game files.
2. Ensure the game format matches the expectations of the `Solver` classes.

---

## **Maintainer**

This repository is maintained by:

**Chris Lai**  
Email: [chrislai_502@berkeley.edu](mailto:chrislai_502@berkeley.edu)
