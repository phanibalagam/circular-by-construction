"""08 — Manuscript integrity gate. Fails the build on any disagreement.

House convention, ported from papers 1 and 10. Run this before packaging or
submitting. It is the authority on the three things a late edit is most likely to
break silently:

  1. Every number in numbers.tex still agrees with results/. Re-derived here from
     the JSON, independently of src/06_tables.py, so a bug in the generator shows
     up as a failure instead of being reproduced.
  2. The programme disclaimer is present and verbatim in the manuscript, and the
     "Independent work." notice is on every .py under src/.
  3. The manuscript still compiles clean and its declared artefacts exist: every
     \\includegraphics target present, every macro used is defined, every cited
     key is in the bibliography and in the .bbl.

Exit status 0 = pass, 1 = fail. Nothing is written; this only reads and reports.

Independent work. Carried out on personal time and equipment, not connected to
the author's employment. No proprietary, confidential or internal data of any
organization was used. All data is public.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

DISCLAIMER = (
    "This work was carried out independently, on personal time and equipment, and is "
    "not connected to the author's employment. The views expressed are the author's "
    "own and do not represent the views, positions or policies of any current, former "
    "or future employer or client. No proprietary, confidential or internal data of "
    "any organization was used."
)
CODE_NOTICE = "Independent work."

failures: list[str] = []
checks = 0


def check(ok: bool, label: str, detail: str = "") -> None:
    global checks
    checks += 1
    if ok:
        print(f"  pass  {label}")
    else:
        print(f"  FAIL  {label}{(' — ' + detail) if detail else ''}")
        failures.append(label)


# --------------------------------------------------------------------------
def check_numbers() -> None:
    """Re-derive every macro from results/ and compare with numbers.tex."""
    print("\n[1] numbers.tex against results/")
    sys.argv = ["verify"]
    trace = ROOT / "selfcheck" / "trace_numbers.py"
    if not trace.exists():
        check(False, "number trace available", f"{trace} missing")
        return
    r = subprocess.run([sys.executable, str(trace)], cwd=trace.parent,
                       capture_output=True, text=True)
    out = (r.stdout or "") + (r.stderr or "")
    m = re.search(r"macros=(\d+) match=(\d+) mismatch=(\d+) notchecked=(\d+)", out)
    if not m:
        check(False, "number trace ran", out.strip()[-300:])
        return
    total, match, mismatch, notchecked = (int(g) for g in m.groups())
    check(mismatch == 0, f"all {total} macros agree with results/",
          f"{mismatch} mismatched")
    check(notchecked == 0, "every macro independently re-derived",
          f"{notchecked} not checked")
    undef = re.search(r"undefined=(\d+)", out)
    if undef:
        check(int(undef.group(1)) == 0, "no macro used but undefined")


def check_disclaimer() -> None:
    print("\n[2] disclaimer and code notices")
    tex = (ROOT / "paper.tex").read_text(encoding="utf-8")
    flat = re.sub(r"\s+", " ", tex)
    check(re.sub(r"\s+", " ", DISCLAIMER) in flat,
          "programme disclaimer present and verbatim in paper.tex")
    check("organisation" not in tex,
          "house spelling 'organization' used throughout paper.tex")
    missing = [p.name for p in sorted((ROOT / "src").glob("*.py"))
               if CODE_NOTICE not in p.read_text(encoding="utf-8")]
    check(not missing, "every src/*.py carries the Independent work notice",
          ", ".join(missing))
    ai = "Declaration of generative AI and AI-assisted technologies"
    check(ai in tex, "generative-AI declaration present (arXiv moderation policy)")


def check_manuscript() -> None:
    print("\n[3] manuscript artefacts")
    tex = (ROOT / "paper.tex").read_text(encoding="utf-8")

    figs = re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", tex)
    absent = [g for g in figs if not (ROOT / "figures" / g).exists()]
    check(not absent, f"all {len(figs)} figure files present", ", ".join(absent))

    labels = set(re.findall(r"\\label\{([^}]+)\}", tex))
    refs = set(re.findall(r"\\ref\{([^}]+)\}", tex))
    check(not (refs - labels), "no dangling \\ref", ", ".join(sorted(refs - labels)))

    defined = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}",
                             (ROOT / "numbers.tex").read_text(encoding="utf-8")))
    defined |= set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", tex))
    builtin = {"linewidth", "textemdash", "centering", "maketitle", "today", "rho",
               "pm", "checkmark", "star", "dagger", "textbf", "noindent", "times",
               "footnotesize", "scriptsize", "S", "emph", "url"}
    used = set(re.findall(r"\\([A-Za-z]+)\{\}", tex))
    undefined = sorted(used - defined - builtin)
    check(not undefined, "every macro used in paper.tex is defined",
          ", ".join(undefined))

    cited: set[str] = set()
    for m in re.findall(r"\\cite[a-z]*\{([^}]+)\}", tex):
        cited |= {c.strip() for c in m.split(",")}
    bib = set(re.findall(r"@\w+\{([^,]+),",
                         (ROOT / "references.bib").read_text(encoding="utf-8")))
    check(not (cited - bib), "every cited key is in references.bib",
          ", ".join(sorted(cited - bib)))
    bbl_path = ROOT / "paper.bbl"
    if bbl_path.exists():
        bbl = set(re.findall(r"\\bibitem\[[^\]]*\]\{([^}]+)\}",
                             bbl_path.read_text(encoding="utf-8")))
        check(not (cited - bbl), "every cited key is in paper.bbl (arXiv runs no BibTeX)",
              ", ".join(sorted(cited - bbl)))
    check(not (bib - cited), "no uncited entry left in references.bib",
          ", ".join(sorted(bib - cited)))


def check_pdf() -> None:
    print("\n[4] compiled PDF")
    pdf = ROOT / "paper.pdf"
    if not pdf.exists():
        check(False, "paper.pdf exists", "build it first")
        return
    try:
        out = subprocess.run(["pdffonts", str(pdf)], capture_output=True,
                             text=True).stdout
        rows = out.splitlines()[2:]
        check(sum("Type 3" in r for r in rows) == 0, "no Type 3 fonts")
        check(sum(1 for r in rows if r.split()[3:4] == ["no"]) == 0,
              "all fonts embedded")
    except FileNotFoundError:
        print("  skip  pdffonts not available")
    try:
        txt = subprocess.run(["pdftotext", "-f", "1", "-l", "1", str(pdf), "-"],
                             capture_output=True, text=True).stdout
        m = re.search(r"Abstract\s(.*?)This work was carried out", txt, re.S)
        if m:
            n = len(re.sub(r"\s+", " ", m.group(1)).strip())
            check(n <= 1920, f"abstract {n} characters within arXiv's 1920 limit")
    except FileNotFoundError:
        print("  skip  pdftotext not available")


def main() -> int:
    print("Manuscript integrity gate — paper 8")
    check_numbers()
    check_disclaimer()
    check_manuscript()
    check_pdf()
    print(f"\n{checks - len(failures)}/{checks} checks passed")
    if failures:
        print("FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
