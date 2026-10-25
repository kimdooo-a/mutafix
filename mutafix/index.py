"""Function index and candidate retrieval for a Python repository.

Pre-existing component: this module is adapted from the MutaLoc fault-localization
benchmark code (October 2026, same author). It provides
  * an AST index of every function and method in a package (nested ones included),
  * BM25 over function source,
  * a static call graph walked breadth-first from the names a failing test mentions,
    with two implicit edge kinds (a class reaches its dunder methods, a function reaches
    the functions nested inside it).
MutaFix uses it to produce a short candidate list; the language model never sees the
whole repository.
"""
import ast
import math
import os
import re
from collections import Counter, defaultdict

TOK_RE = re.compile(r"[A-Za-z][a-z]+|[A-Z]+(?![a-z])|\d+")
STOP = set(["self", "cls", "def", "return", "if", "else", "elif", "for", "in", "not", "and", "or", "is", "none", "true", "false", "import", "from", "as", "with", "while", "try", "except", "raise", "pass", "the", "to", "of", "a", "an", "it", "be", "this", "that", "lambda", "yield", "assert"])


def functions_in(tree):
    """Yield (qualname, node) for every function/method, including nested ones."""
    out = []

    def walk(node, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                q = f"{prefix}{child.name}"
                out.append((q, child))
                walk(child, q + ".<locals>.")
            elif isinstance(child, ast.ClassDef):
                walk(child, f"{prefix}{child.name}.")
            else:
                walk(child, prefix)
    walk(tree, "")
    return out


def own_nodes(fn):
    """Nodes that belong to fn's body but not to a nested function/class."""
    stack = list(fn.body)
    while stack:
        n = stack.pop()
        yield n
        for c in ast.iter_child_nodes(n):
            if isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                continue
            stack.append(c)


def tokenize(text):
    toks = []
    for w in re.split(r"[^A-Za-z0-9]+", text):
        if not w:
            continue
        parts = [p.lower() for p in TOK_RE.findall(w)]
        for p in parts + ([w.lower()] if len(parts) > 1 else []):
            if len(p) >= 2 and p not in STOP:
                toks.append(p)
    return toks


class BM25:
    def __init__(self, docs, k1=1.2, b=0.75):
        self.docs = [Counter(d) for d in docs]
        self.len = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.len) / max(1, len(self.len))
        df = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.k1, self.b = k1, b

    def scores(self, q):
        qc = Counter(q)
        out = []
        for d, L in zip(self.docs, self.len):
            s = 0.0
            for t in qc:
                f = d.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * L / self.avg))
            out.append(s)
        return out


def called_names(nodes):
    names = set()
    for n in nodes:
        if isinstance(n, ast.Name):
            names.add(n.id)
        elif isinstance(n, ast.Attribute):
            names.add(n.attr)
    return names


class RepoIndex:
    """Index of the functions under <root>/<package>, built from the current files on disk."""

    def __init__(self, root, package):
        self.root = root
        self.funcs = []
        pkg = os.path.join(root, package)
        for dp, dn, fns in os.walk(pkg):
            dn[:] = sorted(d for d in dn if d not in ("tests", "__pycache__"))
            for f in sorted(fns):
                if not f.endswith(".py") or f.startswith("test_"):
                    continue
                path = os.path.join(dp, f)
                rel = os.path.relpath(path, root).replace("\\", "/")
                src = open(path, encoding="utf-8").read()
                lines = src.splitlines()
                try:
                    tree = ast.parse(src)
                except SyntaxError:
                    continue
                for q, fn in functions_in(tree):
                    start = fn.decorator_list[0].lineno if fn.decorator_list else fn.lineno
                    self.funcs.append({
                        "id": f"{rel}::{q}", "file": rel, "qualname": q, "short": fn.name,
                        "start": start, "lineno": fn.lineno, "end_lineno": fn.end_lineno,
                        "code": "\n".join(lines[start - 1:fn.end_lineno]),
                        "calls": called_names(own_nodes(fn)),
                    })
        self.by_short = defaultdict(list)
        for i, f in enumerate(self.funcs):
            if "<locals>" not in f["qualname"]:
                self.by_short[f["short"]].append(i)
        self.class_dunders = defaultdict(list)
        self.nested = defaultdict(list)
        for i, f in enumerate(self.funcs):
            q = f["qualname"]
            if "<locals>" in q:
                self.nested[(f["file"], q.rsplit(".<locals>.", 1)[0])].append(i)
            elif "." in q and f["short"].startswith("__") and f["short"].endswith("__"):
                self.class_dunders[q.rsplit(".", 1)[0].split(".")[-1]].append(i)
        self.bm25 = BM25([tokenize(f["code"]) for f in self.funcs])

    def find(self, file, line):
        """Innermost indexed function whose span contains file:line."""
        best = None
        for i, f in enumerate(self.funcs):
            if f["file"] == file and f["start"] <= line <= f["end_lineno"]:
                if best is None or f["start"] >= self.funcs[best]["start"]:
                    best = i
        return best

    def test_source(self, nodeid):
        """Source of the test function a pytest node id points to, and the names it uses."""
        parts = nodeid.split("::")
        path = os.path.join(self.root, parts[0])
        if not os.path.exists(path):
            return "", set()
        src = open(path, encoding="utf-8").read()
        tree = ast.parse(src)
        lines = src.splitlines()
        names = [re.sub(r"\[.*$", "", p) for p in parts[1:]]
        node, scope = None, tree
        for nm in names:
            found = None
            for c in ast.walk(scope) if scope is tree else ast.iter_child_nodes(scope):
                if isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and c.name == nm:
                    found = c
                    break
            if found is None:
                break
            node = scope = found
        if node is None or isinstance(node, ast.ClassDef):
            return "", set()
        start = node.decorator_list[0].lineno if node.decorator_list else node.lineno
        return "\n".join(lines[start - 1:node.end_lineno]), called_names(ast.walk(node))

    def bm25_rank(self, query_text):
        s = self.bm25.scores(tokenize(query_text))
        return sorted(range(len(s)), key=lambda i: (-s[i], i)), s

    def _resolve(self, name):
        return list(self.by_short.get(name, [])) + self.class_dunders.get(name, [])

    def _closure(self, i):
        res, stack = [i], [i]
        while stack:
            f = self.funcs[stack.pop()]
            for j in self.nested.get((f["file"], f["qualname"]), []):
                res.append(j)
                stack.append(j)
        return res

    def graph_rank(self, test_names, bm25_scores, max_depth=3):
        """BFS over name-resolved call edges from the failing test, BM25 breaks ties."""
        depth, frontier = {}, set()
        for nm in test_names:
            for i0 in self._resolve(nm):
                for i in self._closure(i0):
                    if i not in depth:
                        depth[i] = 1
                        frontier.add(i)
        d = 1
        while frontier and d < max_depth:
            d += 1
            nxt = set()
            for i in frontier:
                for nm in self.funcs[i]["calls"]:
                    for j0 in self._resolve(nm):
                        for j in self._closure(j0):
                            if j not in depth:
                                depth[j] = d
                                nxt.add(j)
            frontier = nxt
        inf = max_depth + 1
        order = sorted(range(len(self.funcs)), key=lambda i: (depth.get(i, inf), -bm25_scores[i], i))
        return order, depth
