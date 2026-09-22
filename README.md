# Artificial intelligence adoption and strategic governance in universities

This repository contains the data, Stata code, and supporting documentation for the study:

**Artificial Intelligence Adoption and Strategic Governance in Universities: The Roles of Organizational Learning and Digital Capability**

## Project overview

The study examines whether the adoption of artificial intelligence is associated with stronger strategic governance in higher education institutions. It evaluates two organisational pathways, organisational learning and digital capability, and compares the relationship across research intensive, comprehensive or teaching oriented, and applied or vocational institutions.

The empirical analysis uses an institution level panel of 300 Chinese higher education institutions observed annually from 2020 through 2025. The panel contains 1,800 institution year observations. The study reports pooled ordinary least squares, institution fixed effects, random effects, mediation, moderation, institutional type comparisons, and robustness analyses.

The four main constructs are measured with composite indices: AI adoption, strategic governance, organisational learning, and digital capability. The analysis also includes university scale, financial investment intensity, faculty qualification ratio, province, year, and institutional type.

## Repository contents

- `data.csv`: analysis ready panel data used in the reported models.
- `analysis.do`: Stata code for data import, descriptive statistics, regression models, mediation, moderation, institutional type analyses, robustness checks, and figures.
- `README_reproducibility.md`: file descriptions and reproduction notes.

## Data and sources

The variables were compiled from institutional informatization annual reports, provincial higher education statistical yearbooks, and institutional strategic plan disclosures. The repository is intended to provide the data and code needed to examine the reported analyses. Source records and access information should be added here for any materials that cannot be redistributed.

## Reproduction

Open Stata in a working directory containing `data.csv` and `analysis.do`, then run the do file. The script reads `data.csv` from the same directory and generates the reported analyses and figures. The code uses Stata version 17 syntax.

## Citation

Please cite the associated manuscript when using these materials. The repository is maintained for peer review and research transparency.
