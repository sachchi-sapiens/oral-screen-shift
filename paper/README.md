# Paper

`main.tex` is the manuscript, `references.bib` the references, `figures/` the three figures (copied from `results/`). It compiles with pdfLaTeX and BibTeX.

## Compile on Overleaf

1. Overleaf > New project > Upload project > choose `oral-screen-shift-paper.zip` (in this folder, or zip this folder yourself).
2. Menu > Compiler: pdfLaTeX. Main document: `main.tex`.
3. Recompile. Check that the two wide tables fit the page and that every reference resolves.

## Where to submit, in order

| Step | Venue | Cost | Why |
|---|---|---|---|
| 1 | arXiv (`cs.CV`, cross-list `eess.IV`) | free | A public, citable record within a few days. Allowed by every venue below. |
| 2 | Transactions on Machine Learning Research (TMLR) | free, no fees of any kind | Accepts papers whose claims are supported by evidence and that some readers would find useful; novelty of method is not required. A careful empirical study with negative results fits. Rolling submission. |
| Backup | MIDL 2027 short paper track (Porto, July 2027) | free to submit, registration fee to attend | 3 pages. The 2026 deadline was 15 April; 2027 dates are not yet announced. |
| Backup | Machine Learning for Biomedical Imaging (MELBA) | described as a minimal fee | Journal-length papers, arXiv overlay. Check the current fee before submitting. |

Avoid journals that email you offering fast acceptance for a fee.

## Before submitting to TMLR

- TMLR is double-blind. Use the official TMLR LaTeX template on Overleaf, paste the body of `main.tex` into it, and remove the author block and the GitHub link (replace it with an anonymised link such as anonymous.4open.science, or "available after review").
- The arXiv version may keep your name; TMLR allows preprints.
- Read the paper once in full and check every number against `results/*.csv`.
- Ask one person (a professor, or a mentor at work) to read it before you submit.
