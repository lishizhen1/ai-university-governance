# AI adoption and university governance

This repository contains the data and analysis code for the manuscript:

**Artificial Intelligence Adoption and Strategic Governance in Universities: The Roles of Organizational Learning and Digital Capability**

## Contents

- `data.csv`: analysis-ready university-year panel data for 300 institutions observed from 2020 through 2025 (1,800 institution-year observations).
- `analysis.do`: Stata code for the descriptive statistics, regressions, mediation analysis, moderation analysis, heterogeneity analysis, robustness checks, and figures reported in the manuscript.

## Variables

The panel includes the university identifier, year, province, university type, AI adoption, strategic governance, organizational learning, digital capability, university scale, financial investment, and faculty qualification ratio.

University type is coded as 1 = research-intensive, 2 = comprehensive/teaching, and 3 = applied/vocational.

## Reproduction

Open Stata in a working directory containing the data file and the do-file. The do-file reads `hssc_ai_university_panel_data.csv` from the same working directory. The script documents the model specifications and generates the analysis figures and tables used in the manuscript.

## Data sources and access

The variables were compiled from institutional informatization annual reports, provincial higher education statistical yearbooks, and institutional strategic-plan disclosures. Source records and access information should be documented here before the repository is submitted to the journal. Any source material that cannot be redistributed should be identified with the applicable access restriction.

