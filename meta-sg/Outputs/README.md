# Outputs

- [archive](archive/manifest.json): historical document figures, result tables,
  audits and available underlying source metrics. The root [Meta-SG.md](../Meta-SG.md)
  links here; its [original Obsidian source](archive/Meta-SG.obsidian.md) is unchanged.
- [quickstart](quickstart/README.md): measured results and figures from the new
  MNIST integration demo. See [Quickstart](../QUICKSTART.md) to reproduce it.
- `runs/`: new execution artifacts, including checkpoints. Ignored by Git.

Historical CSV/JSON/image/PDF/SVG files are preserved byte for byte. Markdown
resource links are rewritten for GitHub. The archive manifest records source
paths, source/stored SHA-256 hashes and file sizes, including source metrics
collected under `archive/source-data/`. Absolute paths inside historical tables
and audits remain provenance fields; use the manifest to locate archived copies.

Only curated `archive/` and `quickstart/` assets are tracked. Dataset downloads,
normal run directories and new model checkpoints stay outside Git.
