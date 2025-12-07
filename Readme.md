# 🌟 Automata Construction Tool: Regex-to-DFA Simulator

## Project Overview

This is a comprehensive desktop application developed in Python using the **Tkinter** library for the graphical user interface (GUI). It serves as a visual and functional demonstration of key concepts in the Theory of Automata, specifically focusing on the conversion pipeline from a Regular Expression (Regex) to a Minimized Deterministic Finite Automaton (Minimized DFA).

The core alphabet for this simulator is **{n, o, p}**.

---

👥 Team Members Shahroz Khalid [53531]

Sinwan Haider [56275]

📚 Academic Context Course: Theory of Automata Section: BSCS 5-1 Faculty: Computing Supervisor: Dr. Musharraf Ahmed Submission Date: 17 November 2025

📄 License This project is licensed for academic purposes under the Faculty of Computing, Theory of Automata Course.

🆘 Support For questions or issues regarding this project, please contact:

Shahroz Khalid: [53531@students.riphah.edu.pk]

Sinwan Haider: [56275@student.riphah.edu.pk]

Note: This project is developed as part of the Theory of Automata course requirements at the Faculty of Computing.


## ⚙️ Core Functionality and Methods Used

The application provides sequential steps for automata construction, with the specific theoretical method cited for each conversion:

| Step | Functionality | Theoretical Method Implemented |
| :--- | :--- | :--- |
| **1. Build NFA** | Converts the input Regular Expression into an NFA. | **Thompson's Construction Algorithm** |
| **2. Build DFA** | Converts the NFA (Non-deterministic) into an equivalent DFA (Deterministic). | **The Subset Construction Method** |
| **3. Minimize DFA** | Reduces the DFA to the smallest equivalent DFA. | **The Table Filling Method** (or Myhill-Nerode Theorem application) |
| **4. Simulate** | Accepts an input string and checks for acceptance/rejection using the Minimized DFA, logging the state transitions. | **DFA Simulation Algorithm** |

---

## 🚀 How to Run the Project

### Prerequisites

1.  **Python:** Ensure you have Python 3.x installed.
2.  **Required Libraries:** The project uses `tkinter` (usually built-in), `graphviz` for drawing the automata diagrams (`.png` files), and standard Python modules. You must install the `graphviz` Python package and the external `graphviz` system package.

    ```bash
    pip install graphviz
    ```

### Execution

1.  Clone the repository:
    ```bash
    git clone [https://github.com/YourUsername/Regex-to-DFA-Simulator.git](https://github.com/YourUsername/Regex-to-DFA-Simulator.git)
    cd Regex-to-DFA-Simulator
    ```
2.  Run the main application file:
    ```bash
    python main.py
    ```

---

## 🖼️ Visual Results (Screenshots)

The following screenshots illustrate the application's interface and the step-by-step conversion process.

### 1. NFA Construction (Thompson's Method)

Shows the initial NFA built from the default regular expression and the corresponding transition table.


### 2. Minimized DFA Diagram

Shows the final minimized DFA resulting from the conversion pipeline.


---

## 💻 Project Structure

The project is divided into three main files for clean separation of concerns:

| File | Description |
| :--- | :--- |
| `main.py` | The project entry point. Initializes the GUI and includes a quick command-line interface (CLI) test function. |
| `automata_gui.py` | Contains all the **Tkinter** code for the GUI layout, button commands, table population, and output logging. |
| `automata_core.py` | Contains the **core logic** for automata theory: NFA/DFA classes, Regex-to-Postfix conversion, Thompson's construction, Subset construction (`nfa_to_dfa`), Table Filling minimization (`minimize_dfa`), and the simulation algorithm. |
