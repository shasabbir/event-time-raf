# Reusable Prompt — "Rewrite Methodology in Sample-Journal Style + Figures + Full Paper Sync"

Copy everything inside the block below into a new chat, replace the `[[ ]]` placeholders,
and attach THREE files:
1. Your paper as a LaTeX project (.zip)
2. The sample/reference journal paper (.pdf)
3. `research_paper_instructions.md` (the assignment/requirements guide)

---

```
I'm attaching THREE files. Read all three before doing anything.

FILE 1 — my paper as a LaTeX project (zip):
    [[SHORT DESCRIPTION, e.g. "self-supervised anomaly detection in financial time series"]]
FILE 2 — a sample high-quality journal paper (PDF) whose Methodology I want to emulate.
FILE 3 — research_paper_instructions.md: the official assignment guide listing every required
    section, the standard expected, and the final submission checklist.

Treat FILE 3 as the specification, FILE 2 as the style/depth reference, and FILE 1 as the
artifact to be upgraded.

=====================================================================
PART A — STUDY THE SAMPLE (FILE 2)
=====================================================================
Read the sample paper's Methodology carefully and extract its *structural template*, not its
content. Specifically identify:
- how it opens (a high-level framing paragraph naming the components),
- its "Pipeline Overview" subsection with a compact tensor-notation forward pass equation,
- how it breaks the model into NAMED MODULAR BLOCKS, each with its own subsection, its own
  figure, and 1-3 numbered equations,
- how every figure is walked through in prose ("as shown in Figure N"),
- its notation conventions (tensor shapes, symbols, equation numbering, caption style).

=====================================================================
PART B — MAP THE SAMPLE'S MODULES ONTO MY DOMAIN
=====================================================================
Do NOT copy the sample's domain content. Re-theme each of its named modules into an equivalent
that is faithful to MY paper's actual architecture and data. Give me the mapping table first
and wait for nothing — just proceed after stating it, e.g.:
    [[SAMPLE MODULE 1]]  ->  [[MY EQUIVALENT MODULE 1]]
    [[SAMPLE MODULE 2]]  ->  [[MY EQUIVALENT MODULE 2]]
    [[SAMPLE MODULE 3]]  ->  [[MY EQUIVALENT MODULE 3]]
Every renamed module must be technically coherent with my data modality
([[e.g. 1-D time series / tabular / image / text]]) — no leftover concepts from the sample's
domain.

HARD CONSTRAINTS
- Do NOT change my reported numbers, results, tables, dataset details, or ablation values.
- PRESERVE VERBATIM all existing methodology equations and algorithm blocks:
  [[LIST THEM, e.g. "contrastive loss, masked-reconstruction loss, joint objective, clustering
  density, detector scores, fusion rule, cascade algorithm"]].
  New material must slot AROUND them without reordering them.
- Keep my LaTeX class, packages, bibliography, and label conventions intact. Reuse existing
  figure labels where a figure is being replaced so in-text \ref's still resolve.
- Match my paper's existing symbol set; define any new symbol on first use.

=====================================================================
PART C — WRITE THE NEW METHODOLOGY
=====================================================================
Produce LaTeX containing:
- A rewritten opening paragraph naming the [[N]] top-level components.
- A "Framework Overview and Forward Pass" subsection with one compact multi-line equation
  giving the end-to-end forward pass in tensor notation, all shapes stated.
- One subsection per named module: motivating paragraph, figure reference with prose
  walkthrough, and numbered equations.
- Formal academic register, third person, no marketing language, no bullet-point padding.
Depth target: a competent reader must be able to reimplement the model from this section alone
(this is an explicit requirement in FILE 3).

=====================================================================
PART D — FIGURES
=====================================================================
Create [[4]] clean, professional VECTOR figures as SVG (plus PDF versions for LaTeX inclusion),
in the same aesthetic as the sample paper's figures:
- rounded pastel-filled blocks with thin dark strokes, grouped dashed containers with titles,
- explicit tensor shapes annotated on arrows/blocks,
- labelled arrows, residual/skip paths as elbow connectors,
- small monospace annotations for layer types and hyperparameters,
- consistent palette and font sizing across all figures, with cross-references between them.
Figures to produce:
    Fig 1 — end-to-end pipeline overview (data -> preprocessing -> encoder -> heads -> decision)
    Fig 2 — [[MY MODULE 1]] internals
    Fig 3 — [[MY MODULE 2]] internals
    Fig 4 — [[MY MODULE 3]] / prediction-head internals
Generate them programmatically (a small reusable Python SVG builder is fine), render previews,
visually verify they are clean, well-spaced and non-overlapping, and iterate until they are
publication-ready. Escape XML special characters properly. No screenshots, no raster diagrams
— FILE 3 explicitly forbids these.

=====================================================================
PART E — FULL-PAPER AUDIT AND SYNC AGAINST research_paper_instructions.md
=====================================================================
This part is as important as the methodology rewrite. After the new Methodology is written,
go back through the ENTIRE paper and check every existing component against FILE 3, then bring
the whole manuscript into consistency with both FILE 3's requirements and the new methodology.

1) Run the FILE 3 submission checklist item by item against my paper. Produce a table:
       Requirement | Status (Present / Weak / Missing) | Where it is | What I changed
   Cover every required section in FILE 3's order: Title, Abstract, Introduction (+bulleted
   contributions), Related Work (thematic sub-headings, comparative not list-like), Critical
   Gaps and Limitations (mandatory), Core Contributions, Methodology, Dataset Description,
   Code Availability, Experimental Setup, Evaluation Metrics, Comparison with Existing Studies,
   Ablation Study (mandatory), Explainability Analysis (if applicable), Results and Discussion,
   Limitations, Future Work, Conclusion, References, Figures/Tables presentation rules.

2) Propagate the new methodology everywhere it is referenced:
   - Abstract: must name the new components and keep the same headline numbers.
   - Introduction contributions bullets: one bullet per named component, phrased as a
     mechanism, not "we improved accuracy".
   - Critical Gaps section: make the gap -> contribution mapping one-to-one and explicit, the
     way FILE 3 describes.
   - Results/Discussion, Ablation, Limitations, Conclusion: update any wording that describes
     the architecture so it uses the new component names and the new forward-pass framing.
   - Hyperparameter/reproducibility table: add rows for every new module hyperparameter.
   - Ablation table: ensure a row exists for each named component so each one is justified;
     if I have no measured number for a newly named module, FLAG IT rather than inventing one.

3) Consistency sweep:
   - No contradictory metric values anywhere (check abstract vs results vs conclusion and
     report any mismatch you find rather than silently picking one).
   - All figures, tables and equations numbered and referenced in body text.
   - Terminology and symbols identical across sections.
   - Reference count and recency against FILE 3's target ([[40-60, mostly 2023-2026, IEEE
     style]]) — report the actual count and year distribution, and flag shortfalls.

4) Anything FILE 3 requires that my paper genuinely lacks and that you cannot fabricate
   (missing dataset link, missing ablation run, missing repo link, missing external validation):
   list it separately as "ACTION REQUIRED FROM ME" with a one-line note on what I need to supply.

=====================================================================
PART F — INTEGRATE AND VERIFY
=====================================================================
- Insert figures with proper \begin{figure*} blocks, captions in the sample paper's caption
  style, and \label's.
- Compile with pdflatex (multiple passes, bibtex if needed); confirm no broken refs, no
  undefined citations, and that all figures render.

=====================================================================
DELIVERABLES
=====================================================================
1. Updated .tex file
2. All figures as .svg AND .pdf
3. One zip with the complete updated project
4. A changelog: exactly what changed and where
5. The FILE 3 checklist audit table from Part E
6. The "ACTION REQUIRED FROM ME" list

FINAL NOTE
Flag clearly, at the end, any module you introduced that is an *enrichment* rather than a
description of code I already have, so I can implement it or drop it before submission. Same
for any claim you strengthened in the abstract or contributions that my results do not yet
support.
```
