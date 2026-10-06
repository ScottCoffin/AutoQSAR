"""Regenerate submission/abstract.tex from the manuscript.md abstract and report its word count."""
import re
from pathlib import Path

md = Path("manuscript.md").read_text(encoding="utf-8")
block = md.split("## Abstract", 1)[1].split("**Keywords:**", 1)[0].strip()
paras = [p.strip() for p in block.split("\n\n") if p.strip()]


def to_tex(s: str) -> str:
    s = s.replace("%", "\\%")
    s = re.sub(r"\+(\d+(?:\.\d+)?)\\%", r"$+\1$\\%", s)
    s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s)
    s = s.replace("model-dataset", "model--dataset")
    return s


Path("submission/abstract.tex").write_text("\n\n".join(to_tex(p) for p in paras) + "\n", encoding="utf-8")
words = len(re.sub(r"\*\*", "", block).split())
sci = paras[-1].split("**Scientific Contribution.**", 1)[1]
print(f"abstract words: {words} (cap 350); scientific-contribution sentences: {len(re.findall(r'[.!?](?:\s|$)', sci.strip()))}")

ms = Path("submission/manuscript.tex")
t = ms.read_text(encoding="utf-8")
old = re.search(r"(\\caption\*\{\\textbf\{Graphical abstract\.\}) (.*?)\}\n", t).group(2)
ga = md.split("**Graphical abstract.**", 1)[1].split("\n", 1)[0].strip()
t = t.replace(old, to_tex(ga))
ms.write_text(t, encoding="utf-8")
print("caption:", to_tex(ga))
