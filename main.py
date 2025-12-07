# main.py
# Entry point: quick CLI test + start GUI

from automata_core import (
    build_nfa_from_regex,
    nfa_to_dfa,
    minimize_dfa,
    nfa_table_str,
    dfa_table_str,
    simulate_dfa,
)
from automata_gui import start_gui

DEFAULT_REGEX = "(no*n+op*o+pn*p)npn"


def quick_test(regex, samples):
    print("=" * 60)
    print("Quick test for regex:", regex)
    try:
        nfa = build_nfa_from_regex(regex)
    except Exception as e:
        print("Failed to build NFA:", e)
        return
    dfa = nfa_to_dfa(nfa)
    min_d = minimize_dfa(dfa)
    print(nfa_table_str(nfa))
    print(dfa_table_str(min_d))
    for s in samples:
        log, acc = simulate_dfa(min_d, s)
        print(f"\nSimulating '{s}': {'ACCEPTED' if acc else 'REJECTED'}")
        for line in log:
            print(line)


if __name__ == "__main__":
    project_regex = DEFAULT_REGEX
    samples = ["nnpn", "nnnpn", "nonnpn", "ppnpn", "oonpn".replace(" ", ""), "npn"]
    samples = [s for s in samples]

    try:
        quick_test(project_regex, samples)
    except Exception as e:
        print("Quick test error:", e)

    start_gui()
