# Public export notes

This is a scoped code export of the final M3S06022 revision. The scientific analysis algorithms and numerical reference results were not re-estimated during publication.

Packaging adjustments:
- Added a historical-workflow notice to the source-definition README; source Python algorithms remain unchanged.
- Removed only the figure authoring tool path and alignment-audit hook; plotting and numerical calculations are unchanged.

- Updated reproduction instructions to distinguish the public code export from the more extensive local verification package.
- Generated a new manifest for this exact public file set.

Excluded categories: participant data/predictions/assignments; fitted model and preprocessing artifacts; fitted coefficient and centroid files; diagnostic-text extracts; per-threshold ROC source arrays; local environment files; reviewer correspondence and unpublished manuscript documents; private paths and local authoring audits; bundled third-party authoring tools.

The historical source README describes the earlier analysis for provenance. Follow `audit_analysis/REPRODUCTION_README.md` for the current revision. The public code export does not assert public deposition of restricted models, a new prospective protocol, a new clinical validation, or a DOI.

## Revision v2 packaging

The v2 addition uses an explicit file allowlist and retains all v1 files. Public copies of the new Python sources resolve the principal analysis under `audit_analysis/`; original frozen sources are untouched. Historical no-publication wording is updated only in the additional aggregate report copies, and the validation script labels its original execution incident as reference history. No scientific method or reference numerical result was changed during packaging. `additional_revision/SOURCE_PROVENANCE.json` records original/public source hashes. R scripts use a user-provided matching environment; no installers, runtime binaries or private model/trajectory files are included. `additional_revision/verify_public_summaries.py` checks only the aggregate public tables. All public text files use UTF-8 and LF line endings before hashing.

The principal AUDIT_METHODS and reproduction README now identify their v1 scope and link to the completed v2 calibration-error and longitudinal latent-class analyses. Historical source definitions remain explicitly historical. No statement that these additions were left unimplemented is retained in the current public documentation.
