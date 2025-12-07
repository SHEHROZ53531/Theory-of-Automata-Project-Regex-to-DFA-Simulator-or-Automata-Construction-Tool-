# automata_core.py
# Core logic: regex -> NFA -> DFA -> Minimized DFA + simulation
# Alphabet: {n, o, p}

from collections import deque

EPSILON = 'ε'
ALPHABET = set(['n', 'o', 'p'])  # allowed symbols


class NFA:
    def __init__(self):
        self.states = set()
        self.start = None
        self.accept = None
        # { state : { symbol : set(dest_states) } }
        self.trans = dict()

    def _ensure_state(self, s):
        if s not in self.states:
            self.states.add(s)

    def add_state(self, state):
        self._ensure_state(state)

    def add_transition(self, from_state, symbol, to_state):
        self._ensure_state(from_state)
        self._ensure_state(to_state)
        if from_state not in self.trans:
            self.trans[from_state] = dict()
        if symbol not in self.trans[from_state]:
            self.trans[from_state][symbol] = set()
        self.trans[from_state][symbol].add(to_state)

    def get_trans(self, state):
        return self.trans.get(state, {})

    def copy(self):
        n = NFA()
        n.states = set(self.states)
        n.start = self.start
        n.accept = self.accept
        n.trans = {
            s: {sym: set(dests) for sym, dests in syms.items()}
            for s, syms in self.trans.items()
        }
        return n


_state_counter = 0


def new_state():
    global _state_counter
    _state_counter += 1
    return f"q{_state_counter}"


def nfa_for_symbol(symbol):
    if symbol not in ALPHABET:
        raise ValueError(f"Unsupported symbol '{symbol}'. Allowed: {ALPHABET}")
    nfa = NFA()
    s = new_state()
    e = new_state()
    nfa.start = s
    nfa.accept = e
    nfa.add_state(s)
    nfa.add_state(e)
    nfa.add_transition(s, symbol, e)
    return nfa


def nfa_concat(n1, n2):
    nfa = NFA()
    # copy n1
    for st in n1.states:
        nfa.states.add(st)
    for st, syms in n1.trans.items():
        nfa.trans[st] = {sym: set(dests) for sym, dests in syms.items()}
    # copy n2
    for st in n2.states:
        nfa.states.add(st)
    for st, syms in n2.trans.items():
        if st in nfa.trans:
            for sym, dests in syms.items():
                nfa.trans[st].setdefault(sym, set()).update(dests)
        else:
            nfa.trans[st] = {sym: set(dests) for sym, dests in syms.items()}
    nfa.start = n1.start
    nfa.accept = n2.accept
    nfa.add_transition(n1.accept, EPSILON, n2.start)
    return nfa


def nfa_union(n1, n2):
    nfa = NFA()
    s = new_state()
    e = new_state()
    nfa.start = s
    nfa.accept = e
    nfa.add_state(s)
    nfa.add_state(e)
    # copy n1
    for st in n1.states:
        nfa.states.add(st)
    for st, syms in n1.trans.items():
        nfa.trans[st] = {sym: set(dests) for sym, dests in syms.items()}
    # copy n2
    for st in n2.states:
        nfa.states.add(st)
    for st, syms in n2.trans.items():
        if st in nfa.trans:
            for sym, dests in syms.items():
                nfa.trans[st].setdefault(sym, set()).update(dests)
        else:
            nfa.trans[st] = {sym: set(dests) for sym, dests in syms.items()}
    # epsilon connections
    nfa.add_transition(s, EPSILON, n1.start)
    nfa.add_transition(s, EPSILON, n2.start)
    nfa.add_transition(n1.accept, EPSILON, e)
    nfa.add_transition(n2.accept, EPSILON, e)
    return nfa


def nfa_star(n1):
    nfa = NFA()
    s = new_state()
    e = new_state()
    nfa.start = s
    nfa.accept = e
    nfa.add_state(s)
    nfa.add_state(e)
    # copy n1
    for st in n1.states:
        nfa.states.add(st)
    for st, syms in n1.trans.items():
        nfa.trans[st] = {sym: set(dests) for sym, dests in syms.items()}
    # epsilon connections
    nfa.add_transition(s, EPSILON, n1.start)
    nfa.add_transition(s, EPSILON, e)
    nfa.add_transition(n1.accept, EPSILON, n1.start)
    nfa.add_transition(n1.accept, EPSILON, e)
    return nfa


# ---------- Regex -> Postfix ----------

def tokenize_regex(regex):
    s = ''.join(regex.split())
    tokens = []
    i = 0
    while i < len(s):
        c = s[i]
        if c in ('(', ')', '+', '*', '.'):
            tokens.append(c)
            i += 1
        else:
            if c not in ALPHABET:
                raise ValueError(
                    f"Unsupported character '{c}' in regex. "
                    f"Allowed symbols: {ALPHABET} and operators (+,*,(),.)"
                )
            tokens.append(c)
            i += 1
    return tokens


def insert_concats(tokens):
    out = []
    for i, t in enumerate(tokens):
        out.append(t)
        if i + 1 < len(tokens):
            a = t
            b = tokens[i + 1]
            a_is_symbol = a in ALPHABET
            b_is_symbol = b in ALPHABET
            if (a_is_symbol or a == ')' or a == '*') and (b_is_symbol or b == '('):
                out.append('.')
    return out


def tokens_to_postfix(tokens):
    prec = {'*': 3, '.': 2, '+': 1}
    assoc = {'*': 'right', '.': 'left', '+': 'left'}
    stack = []
    out = []
    for tok in tokens:
        if tok in ALPHABET:
            out.append(tok)
        elif tok == '(':
            stack.append(tok)
        elif tok == ')':
            while stack and stack[-1] != '(':
                out.append(stack.pop())
            if not stack:
                raise ValueError("Parentheses mismatched")
            stack.pop()
        else:
            while stack and stack[-1] != '(':
                top = stack[-1]
                if (prec[top] > prec[tok]) or (prec[top] == prec[tok] and assoc[tok] == 'left'):
                    out.append(stack.pop())
                else:
                    break
            stack.append(tok)
    while stack:
        top = stack.pop()
        if top in ('(', ')'):
            raise ValueError("Parentheses mismatched")
        out.append(top)
    return out


def regex_to_postfix(regex):
    tokens = tokenize_regex(regex)
    tokens = insert_concats(tokens)
    postfix = tokens_to_postfix(tokens)
    return ''.join(postfix)


# ---------- Postfix -> NFA (Thompson) ----------

def postfix_to_nfa(postfix):
    stack = []
    for c in postfix:
        if c in ALPHABET:
            stack.append(nfa_for_symbol(c))
        elif c == '*':
            if not stack:
                raise ValueError("Star operator with empty operand")
            n = stack.pop()
            stack.append(nfa_star(n))
        elif c == '.':
            if len(stack) < 2:
                raise ValueError("Concat operator with insufficient operands")
            n2 = stack.pop()
            n1 = stack.pop()
            stack.append(nfa_concat(n1, n2))
        elif c == '+':
            if len(stack) < 2:
                raise ValueError("Union operator with insufficient operands")
            n2 = stack.pop()
            n1 = stack.pop()
            stack.append(nfa_union(n1, n2))
        else:
            raise ValueError(f"Unsupported postfix token: {c}")
    if len(stack) != 1:
        raise ValueError("Invalid postfix expression: leftover stack")
    return stack[0]


def build_nfa_from_regex(regex):
    global _state_counter
    _state_counter = 0
    postfix = regex_to_postfix(regex)
    nfa = postfix_to_nfa(postfix)
    return nfa


# ---------- epsilon-closure & move ----------

def epsilon_closure(nfa, states):
    closure = set(states)
    stack = list(states)
    while stack:
        s = stack.pop()
        for t in nfa.get_trans(s).get(EPSILON, set()):
            if t not in closure:
                closure.add(t)
                stack.append(t)
    return closure


def move(nfa, states, symbol):
    result = set()
    for s in states:
        dests = nfa.get_trans(s).get(symbol, set())
        result.update(dests)
    return result


# ---------- DFA + conversion ----------

class DFA:
    def __init__(self):
        self.states = set()      # set of frozensets
        self.start = None        # frozenset
        self.accepts = set()     # subset of states
        self.trans = dict()      # { state : { symbol : state } }
        self.symbols = set()


def nfa_to_dfa(nfa):
    dfa = DFA()
    syms = set()
    for st, syms_map in nfa.trans.items():
        for sym in syms_map.keys():
            if sym != EPSILON:
                syms.add(sym)
    dfa.symbols = syms

    start_closure = frozenset(epsilon_closure(nfa, {nfa.start}))
    dfa.start = start_closure
    dfa.states.add(start_closure)
    queue = deque([start_closure])
    dfa.trans[start_closure] = dict()

    while queue:
        cur = queue.popleft()
        dfa.trans.setdefault(cur, {})
        for sym in syms:
            mv = move(nfa, cur, sym)
            if not mv:
                continue
            nxt = frozenset(epsilon_closure(nfa, mv))
            dfa.trans[cur][sym] = nxt
            if nxt not in dfa.states:
                dfa.states.add(nxt)
                queue.append(nxt)
                dfa.trans.setdefault(nxt, {})

    for st in dfa.states:
        if nfa.accept in st:
            dfa.accepts.add(st)
    return dfa


# ---------- DFA minimization ----------

def minimize_dfa(dfa):
    states = list(dfa.states)
    n = len(states)
    if n == 0:
        return DFA()

    index = {s: i for i, s in enumerate(states)}
    table = [[False] * n for _ in range(n)]

    # mark accept vs non-accept
    for i in range(n):
        for j in range(i + 1, n):
            if (states[i] in dfa.accepts) != (states[j] in dfa.accepts):
                table[i][j] = True

    changed = True
    while changed:
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                if table[i][j]:
                    continue
                for sym in dfa.symbols:
                    t1 = dfa.trans.get(states[i], {}).get(sym)
                    t2 = dfa.trans.get(states[j], {}).get(sym)
                    if t1 is None or t2 is None:
                        if t1 != t2:
                            table[i][j] = True
                            changed = True
                            break
                        else:
                            continue
                    x, y = index[t1], index[t2]
                    if x == y:
                        continue
                    a, b = (x, y) if x < y else (y, x)
                    if table[a][b]:
                        table[i][j] = True
                        changed = True
                        break

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if not table[i][j]:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri

    classes = dict()
    for i in range(n):
        r = find(i)
        classes.setdefault(r, []).append(states[i])

    min_dfa = DFA()
    rep_map = {}
    for rep, members in classes.items():
        merged = frozenset().union(*members)
        rep_map[rep] = merged
        min_dfa.states.add(merged)

    orig_to_merged = {}
    for rep, members in classes.items():
        merged = rep_map[rep]
        for m in members:
            orig_to_merged[m] = merged

    min_dfa.start = orig_to_merged[dfa.start]
    for s in dfa.accepts:
        min_dfa.accepts.add(orig_to_merged[s])

    min_dfa.symbols = set(dfa.symbols)
    min_dfa.trans = {st: {} for st in min_dfa.states}

    for s in dfa.states:
        msrc = orig_to_merged[s]
        for a, dest in dfa.trans.get(s, {}).items():
            mdest = orig_to_merged[dest]
            min_dfa.trans[msrc][a] = mdest

    return min_dfa


# ---------- Pretty tables & simulation ----------

def nfa_table_str(nfa):
    out = []
    out.append("NFA TRANSITION TABLE")
    out.append(f"(ε shown as '{EPSILON}')")
    out.append("")

    states = sorted(nfa.states)
    symbols = sorted({sym for syms in nfa.trans.values() for sym in syms.keys()})

    col_w_state = 8
    col_w = 12

    header = "State".ljust(col_w_state) + "".join(sym.center(col_w) for sym in symbols)
    out.append(header)
    out.append("-" * len(header))

    for s in states:
        row = s.ljust(col_w_state)
        for sym in symbols:
            dests = sorted(nfa.get_trans(s).get(sym, []))
            cell = ",".join(dests) if dests else "-"
            row += cell.center(col_w)
        out.append(row)

    out.append("")
    return "\n".join(out)


def dfa_table_str(dfa):
    out = []
    out.append("DFA TRANSITION TABLE")
    out.append("")

    syms = sorted(dfa.symbols)

    def label_of(s):
        return "{" + ",".join(sorted(s)) + "}"

    col_w_state = 26
    col_w = 16

    header = "State".ljust(col_w_state) + "".join(sym.center(col_w) for sym in syms)
    out.append(header)
    out.append("-" * len(header))

    for st in sorted(dfa.states, key=lambda s: (len(s), sorted(s))):
        label = label_of(st)
        row = label.ljust(col_w_state)
        for a in syms:
            nxt = dfa.trans.get(st, {}).get(a)
            cell = label_of(nxt) if nxt else "-"
            row += cell.center(col_w)
        out.append(row)

    out.append("")
    return "\n".join(out)


def format_state(st):
    if isinstance(st, frozenset):
        return "{" + ",".join(sorted(st)) + "}"
    return str(st)


def simulate_dfa(dfa, input_str):
    log = []
    state = dfa.start
    log.append(f"Start at {format_state(state)}")
    step = 0
    for ch in input_str:
        step += 1
        if ch not in dfa.symbols:
            log.append(f"[Step {step}] {format_state(state)} --{ch}--> NO TRANSITION (symbol not in alphabet)")
            return log, False
        nxt = dfa.trans.get(state, {}).get(ch)
        if nxt is None:
            log.append(f"[Step {step}] {format_state(state)} --{ch}--> NO TRANSITION")
            return log, False
        log.append(f"[Step {step}] {format_state(state)} --{ch}--> {format_state(nxt)}")
        state = nxt
    if state in dfa.accepts:
        log.append(f"FINAL STATE REACHED: {format_state(state)}")
        return log, True
    else:
        log.append(f"STOPPED AT NON-FINAL STATE: {format_state(state)}")
        return log, False
