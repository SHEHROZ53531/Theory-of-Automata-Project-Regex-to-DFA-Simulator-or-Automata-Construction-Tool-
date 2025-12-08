# automata_gui.py
# Tkinter GUI for Regex Simulator (uses automata_core)

import os
import webbrowser
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import graphviz

from automata_core import (
    EPSILON,
    ALPHABET,
    build_nfa_from_regex,
    nfa_to_dfa,
    minimize_dfa,
    nfa_table_str,
    dfa_table_str,
    simulate_dfa,
)

DEFAULT_REGEX = "(no*n+op*o+pn*p)npn"

GLOBAL_NFA = None
GLOBAL_DFA = None
GLOBAL_MIN_DFA = None

regex_entry = None
input_entry = None
output_box = None
table_view = None   # Treeview for colored table


# ---------- Graph drawing ----------

def draw_graph(automaton, filename, is_dfa=True):
    dot = graphviz.Digraph(
        comment="Automaton",
        format="png",
        graph_attr={"rankdir": "LR", "fontsize": "12"},
        node_attr={"shape": "circle", "fontsize": "10"},
        edge_attr={"fontsize": "9"},
    )

    start_node = "__start__"
    dot.node(start_node, "", shape="point")

    states = getattr(automaton, "states", [])
    accepts = getattr(automaton, "accepts", {getattr(automaton, "accept", None)})
    start = getattr(automaton, "start")

    for s in states:
        label = "{" + ",".join(sorted(s)) + "}" if isinstance(s, frozenset) else str(s)
        shape = "doublecircle" if s in accepts else "circle"
        dot.node(str(label), label, shape=shape)

    start_label = "{" + ",".join(sorted(start)) + "}" if isinstance(start, frozenset) else str(start)
    dot.edge(start_node, str(start_label))

    trans = getattr(automaton, "trans", {})
    for src, dests in trans.items():
        src_label = "{" + ",".join(sorted(src)) + "}" if isinstance(src, frozenset) else str(src)
        for sym, tgt in dests.items():
            if is_dfa:
                tgt_label = "{" + ",".join(sorted(tgt)) + "}" if isinstance(tgt, frozenset) else str(tgt)
                dot.edge(str(src_label), str(tgt_label), label=sym)
            else:
                for tgt_state in tgt:
                    tgt_label = tgt_state
                    dot.edge(str(src_label), str(tgt_label), label=sym)

    dot.render(filename, cleanup=True)


# ---------- Non-interactive helpers ----------

def build_from_entry(regex):
    global GLOBAL_NFA, GLOBAL_DFA, GLOBAL_MIN_DFA
    nfa = build_nfa_from_regex(regex)
    GLOBAL_NFA = nfa
    GLOBAL_DFA = None
    GLOBAL_MIN_DFA = None
    return nfa


def build_dfa_noninteractive():
    global GLOBAL_NFA, GLOBAL_DFA
    if not GLOBAL_NFA:
        return
    GLOBAL_DFA = nfa_to_dfa(GLOBAL_NFA)


def minimize_dfa_noninteractive():
    global GLOBAL_DFA, GLOBAL_MIN_DFA
    if not GLOBAL_DFA:
        return
    GLOBAL_MIN_DFA = minimize_dfa(GLOBAL_DFA)


# ---------- Table population helpers ----------

def populate_table_from_nfa(nfa):
    """Fill Treeview with NFA transitions (State, n, o, p, ε)."""
    if table_view is None:
        return
    table_view.delete(*table_view.get_children())

    states = sorted(nfa.states)
    for i, s in enumerate(states):
        row_vals = []
        for sym in ["n", "o", "p", EPSILON]:
            dests = sorted(nfa.get_trans(s).get(sym, []))
            cell = "{" + ",".join(dests) + "}" if dests else "∅"
            row_vals.append(cell)
        tag = "even" if i % 2 == 0 else "odd"
        table_view.insert("", "end", values=(s, *row_vals), tags=(tag,))

    table_view.tag_configure("even", background="#ffffff")
    table_view.tag_configure("odd", background="#f6f7f9")


def populate_table_from_dfa(dfa):
    """Fill Treeview with DFA transitions (State, n, o, p, ε empty)."""
    if table_view is None:
        return
    table_view.delete(*table_view.get_children())

    syms = ["n", "o", "p"]
    states = sorted(dfa.states, key=lambda s: (len(s), sorted(s)))

    def label_of(s):
        return "{" + ",".join(sorted(s)) + "}"

    for i, st in enumerate(states):
        row_vals = []
        for sym in syms:
            nxt = dfa.trans.get(st, {}).get(sym)
            cell = label_of(nxt) if nxt else "∅"
            row_vals.append(cell)
        row_vals.append("")  # ε column
        tag = "even" if i % 2 == 0 else "odd"
        table_view.insert("", "end", values=(label_of(st), *row_vals), tags=(tag,))

    table_view.tag_configure("even", background="#ffffff")
    table_view.tag_configure("odd", background="#f6f7f9")


# ---------- GUI callbacks (UPDATED OUTPUT MESSAGES) ----------

def gui_build_nfa():
    global GLOBAL_NFA
    regex = regex_entry.get().strip()
    if not regex:
        regex = DEFAULT_REGEX
        regex_entry.delete(0, tk.END)
        regex_entry.insert(0, regex)
    try:
        nfa = build_from_entry(regex)
    except Exception as e:
        messagebox.showerror("Regex Error", str(e))
        return
    output_box.delete(1.0, tk.END)
    # Output message updated
    output_box.insert(tk.END, f"✔ NFA built from regex: {regex} (Thompson's Construction)\n")
    populate_table_from_nfa(nfa)
    try:
        draw_graph(nfa, "nfa", is_dfa=False)
        output_box.insert(tk.END, "NFA diagram saved as nfa.png\n")
    except Exception as e:
        output_box.insert(tk.END, f"Unable to render NFA (Graphviz issue): {e}\n")


def gui_build_dfa():
    global GLOBAL_NFA, GLOBAL_DFA
    if GLOBAL_NFA is None:
        messagebox.showerror("Error", "Please build NFA first.")
        return
    build_dfa_noninteractive()
    output_box.delete(1.0, tk.END)
    # Output message updated
    output_box.insert(tk.END, "✔ DFA built (Subset Construction)\n")
    populate_table_from_dfa(GLOBAL_DFA)
    try:
        draw_graph(GLOBAL_DFA, "dfa", is_dfa=True)
        output_box.insert(tk.END, "DFA diagram saved as dfa.png\n")
    except Exception as e:
        output_box.insert(tk.END, f"Unable to render DFA: {e}\n")


def gui_minimize_dfa():
    global GLOBAL_DFA, GLOBAL_MIN_DFA
    if GLOBAL_DFA is None:
        messagebox.showerror("Error", "Please build DFA first.")
        return
    GLOBAL_MIN_DFA = minimize_dfa(GLOBAL_DFA)
    output_box.delete(1.0, tk.END)
    # Output message updated
    output_box.insert(tk.END, "✔ Minimized DFA (Table Filling Method)\n")
    populate_table_from_dfa(GLOBAL_MIN_DFA)
    try:
        draw_graph(GLOBAL_MIN_DFA, "dfa_min", is_dfa=True)
        output_box.insert(tk.END, "Minimized DFA diagram saved as dfa_min.png\n")
    except Exception as e:
        output_box.insert(tk.END, f"Unable to render minimized DFA: {e}\n")


# ---------- GUI callbacks (UPDATED SIMULATE FUNCTION) ----------

def gui_simulate():
    global GLOBAL_MIN_DFA

    # --- CRITICAL CHANGE: Check if Minimized DFA is built ---
    if GLOBAL_MIN_DFA is None:
        messagebox.showerror(
            "Simulation Error", 
            "The Minimized DFA must be built first. Please click 'Build NFA', 'Build DFA', and 'Minimize DFA'."
        )
        return
    # --------------------------------------------------------

    s = input_entry.get().strip()
    output_box.delete(1.0, tk.END)
    output_box.insert(tk.END, "Simulating on Minimized DFA\n\n")

    populate_table_from_dfa(GLOBAL_MIN_DFA)

    log, accepted = simulate_dfa(GLOBAL_MIN_DFA, s)
    for line in log:
        output_box.insert(tk.END, line + "\n")
    if accepted:
        output_box.insert(tk.END, "\n✔ STRING ACCEPTED\n")
    else:
        output_box.insert(tk.END, "\n✘ STRING REJECTED\n")


def gui_clear():
    output_box.delete(1.0, tk.END)
    if table_view is not None:
        table_view.delete(*table_view.get_children())


def open_img(fn):
    if not fn.lower().endswith(".png"):
        fn += ".png"
    if os.path.exists(fn):
        webbrowser.open(fn)
    else:
        messagebox.showerror("Error", f"{fn} not found. Build first.")


# ---------- GUI layout with colored table (UPDATED BUTTON LABELS) ----------

def start_gui():
    global regex_entry, input_entry, output_box, table_view

    root = tk.Tk()
    root.title("Regex Simulator – Automata Project")
    root.geometry("1050x670")
    root.minsize(1000, 640)

    # Color theme
    BG_WINDOW = "#e3e6ea"
    BG_FRAME = "#ffffff"
    TEXT_MAIN = "#1f2933"
    TEXT_MUTED = "#6b7280"
    HEADER_BG = "#1f3a93"
    HEADER_FG = "#ffffff"

    root.configure(bg=BG_WINDOW)

    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    style.configure("TFrame", background=BG_FRAME)
    style.configure("TLabelFrame", background=BG_FRAME, foreground=TEXT_MAIN)
    style.configure("TLabel", background=BG_FRAME, foreground=TEXT_MAIN, font=("Segoe UI", 9))
    style.configure("Header.TLabel", background=BG_FRAME, foreground=TEXT_MAIN, font=("Segoe UI", 11, "bold"))
    style.configure("Muted.TLabel", background=BG_FRAME, foreground=TEXT_MUTED, font=("Segoe UI", 8))

    style.configure("TButton", padding=4, font=("Segoe UI", 9))
    style.configure("TEntry", fieldbackground="#ffffff", foreground="#000000")

    style.configure(
        "Treeview",
        background="#ffffff",
        foreground="#111111",
        rowheight=22,
        fieldbackground="#ffffff",
        bordercolor="#d0d4da",
        borderwidth=1,
    )
    style.configure(
        "Treeview.Heading",
        background=HEADER_BG,
        foreground=HEADER_FG,
        font=("Segoe UI", 9, "bold"),
        relief="flat",
        bordercolor="#d0d4da",
    )
    style.map("Treeview.Heading", background=[("active", HEADER_BG)])
    style.map("Treeview", background=[("selected", "#cce5ff")], foreground=[("selected", "#111111")])

    # Top bar
    top_outer = tk.Frame(root, bg=BG_WINDOW)
    top_outer.pack(fill="x", pady=(4, 0))

    top_bar = ttk.Frame(top_outer, padding=(10, 8))
    top_bar.pack(fill="x", padx=8)

    ttk.Label(
        top_bar,
        text="Regex Simulator (NFA • DFA • Minimized DFA)",
        style="Header.TLabel",
    ).pack(side="left")

    ttk.Label(
        top_bar,
        text="Alphabet: { n, o, p }    Operators:  +   .   * ( )",
        style="Muted.TLabel",
    ).pack(side="right")

    # Controls strip
    controls_outer = tk.Frame(root, bg=BG_WINDOW)
    controls_outer.pack(fill="x", pady=(4, 0))

    controls = ttk.Frame(controls_outer, padding=(10, 6))
    controls.pack(fill="x", padx=8)

    left_controls = ttk.Frame(controls)
    left_controls.pack(side="left")

    # UPDATED BUTTON LABELS
    ttk.Button(left_controls, text="Build NFA (Thompson's)", command=gui_build_nfa).pack(side="left", padx=3)
    ttk.Button(left_controls, text="Build DFA (Subset)", command=gui_build_dfa).pack(side="left", padx=3)
    ttk.Button(left_controls, text="Minimize DFA (Table Filling)", command=gui_minimize_dfa).pack(side="left", padx=3)
    
    ttk.Button(left_controls, text="Clear Output", command=gui_clear).pack(side="left", padx=10)

    right_controls = ttk.Frame(controls)
    right_controls.pack(side="right")

    ttk.Button(right_controls, text="Open Min DFA", command=lambda: open_img("dfa_min.png")).pack(side="right", padx=3)
    ttk.Button(right_controls, text="Open DFA", command=lambda: open_img("dfa.png")).pack(side="right", padx=3)
    ttk.Button(right_controls, text="Open NFA", command=lambda: open_img("nfa.png")).pack(side="right", padx=3)

    # Regex + Input card
    io_outer = tk.Frame(root, bg=BG_WINDOW)
    io_outer.pack(fill="x", pady=(4, 0))

    io_frame = ttk.Frame(io_outer, padding=(10, 8))
    io_frame.pack(fill="x", padx=8)

    regex_row = ttk.Frame(io_frame)
    regex_row.pack(fill="x", pady=2)

    ttk.Label(regex_row, text="Regular Expression:").pack(side="left")
    regex_entry = ttk.Entry(regex_row)
    regex_entry.pack(side="left", padx=6, fill="x", expand=True)
    regex_entry.insert(0, DEFAULT_REGEX)

    input_row = ttk.Frame(io_frame)
    input_row.pack(fill="x", pady=2)

    ttk.Label(input_row, text="Input String:").pack(side="left")
    input_entry = ttk.Entry(input_row, width=30)
    input_entry.pack(side="left", padx=6)
    ttk.Button(input_row, text="Simulate on Min DFA", command=gui_simulate).pack(side="left", padx=6)

    ttk.Label(
        io_frame,
        text="Tip: Leave regex empty to use default project expression.",
        style="Muted.TLabel",
    ).pack(anchor="w", pady=(4, 0))

    # Sample accepted/rejected examples (what program actually does)
    ttk.Label(
        io_frame,
        text="Accepted examples:  ppnpn   ,   pnpnpn   ,   pnnnpnpn",
        style="Muted.TLabel",
    ).pack(anchor="w")

    ttk.Label(
        io_frame,
        text="Rejected examples:  nnpn   ,   opnpn   ,   oppnpn",
        style="Muted.TLabel",
    ).pack(anchor="w", pady=(0, 2))

    # Output + Table area
    outer = tk.Frame(root, bg=BG_WINDOW)
    outer.pack(fill="both", expand=True, pady=(6, 8))

    # Transition table
    table_frame = ttk.LabelFrame(outer, text="Transition Table", padding=(6, 4))
    table_frame.pack(fill="x", padx=8, pady=(0, 4))

    columns = ("state", "n", "o", "p", "eps")
    table_view = ttk.Treeview(
        table_frame,
        columns=columns,
        show="headings",
        height=7,
    )
    table_view.pack(side="left", fill="x", expand=True)

    table_view.heading("state", text="State")
    table_view.heading("n", text="n")
    table_view.heading("o", text="o")
    table_view.heading("p", text="p")
    table_view.heading("eps", text="ε")

    table_view.column("state", width=90, anchor="center")
    for col in ("n", "o", "p", "eps"):
        table_view.column(col, width=160, anchor="center")

    tbl_scroll = ttk.Scrollbar(table_frame, orient="vertical", command=table_view.yview)
    table_view.configure(yscrollcommand=tbl_scroll.set)
    tbl_scroll.pack(side="right", fill="y")

    # Simulation log
    output_frame = ttk.LabelFrame(outer, text="Simulation Log", padding=(8, 6))
    output_frame.pack(fill="both", expand=True, padx=8, pady=(0, 0))

    output_box = scrolledtext.ScrolledText(
        output_frame,
        width=100,
        height=14,
        font=("Consolas", 10),
        bg="#ffffff",
        fg="#111111",
        insertbackground="#111111",
    )
    output_box.pack(fill="both", expand=True)

    globals()["regex_entry"] = regex_entry
    globals()["input_entry"] = input_entry
    globals()["output_box"] = output_box
    globals()["table_view"] = table_view

    root.mainloop()