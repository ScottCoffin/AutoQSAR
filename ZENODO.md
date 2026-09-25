# Archiving QSARena on Zenodo

The *Journal of Cheminformatics* reproducibility criteria ask for an external archive
(Zenodo or FigShare) referenced from the README, not a bare GitHub link, because
"the likelihood that Zenodo or FigShare disappears is much less" than for a Git host.
This file records how that archive is created and kept current.

**Status: not yet deposited.** The steps below must be run once by the repository owner;
they require a Zenodo login and cannot be automated from this repository alone.

---

## One-time setup (GitHub → Zenodo webhook)

This is the standard route and produces a DOI for every future GitHub release automatically.

1. Sign in at <https://zenodo.org> with the **ScottCoffin** GitHub account
   (*Log in* → *Log in with GitHub*) and authorise the Zenodo application.
2. Go to <https://zenodo.org/account/settings/github/>.
3. Find **ScottCoffin/QSARena** in the repository list and switch the toggle **On**.
   Zenodo now watches the repository for releases.
4. Create a release on GitHub — see below. Zenodo archives the tagged source tree and
   mints a DOI within a few minutes.

Zenodo issues **two** DOIs:

| DOI | Meaning | Use it for |
|---|---|---|
| **Concept DOI** | Always resolves to the newest version | `CITATION.cff`, the README badge, general citation |
| **Version DOI** | Pins one specific release | The paper's *Availability of data and materials* section |

Cite the **version DOI** in the manuscript so reviewers get exactly the code and
artifacts the paper describes.

## Creating the release

```bash
# From a clean working tree on the branch you want archived
git status                      # must be clean
git tag -a v1.0.0 -m "QSARena v1.0.0 — Journal of Cheminformatics submission"
git push origin v1.0.0

gh release create v1.0.0 \
  --title "QSARena v1.0.0" \
  --notes "Release accompanying the Journal of Cheminformatics submission. Includes the
complete benchmark artifacts for the NSF ACCESS Jetstream2 A100 run
(benchmark_results/autoqsar_benchmark_20260623_153839) and the scripts that regenerate
every figure, table and quoted number in the paper."
```

`.zenodo.json` in the repository root controls the archive's title, description, authors,
license and keywords, so the Zenodo record is correct without editing it by hand.

## Including the per-molecule predictions

`predictions.csv` files are excluded from Git by `.gitignore` for size, but the paper's
*Availability of data and materials* section promises them in the archive. The GitHub
webhook only archives what is in the tagged tree, so these must be uploaded separately:

1. Build the supplementary bundle:
   ```bash
   python - <<'PY'
   import tarfile, pathlib
   run = pathlib.Path("benchmark_results/autoqsar_benchmark_20260623_153839")
   out = pathlib.Path("qsarena_predictions_a100.tar.gz")
   with tarfile.open(out, "w:gz") as tar:
       for p in sorted(run.glob("*/predictions.csv")):
           tar.add(p, arcname=str(p.relative_to(run.parent)))
   print("wrote", out, f"{out.stat().st_size/1e6:.1f} MB")
   PY
   ```
2. Open the Zenodo record created by the webhook, choose **New version**, upload the
   tarball alongside the auto-archived source, and publish. Zenodo's default file limit is
   50 GB per record, so size is not a constraint here.

> If the predictions are not present in the local tree (they are gitignored and may have
> been produced on the Jetstream2 instance), retrieve them from that instance before
> building the bundle.

## After minting the DOI

Update these four places, then commit:

1. `README.md` — replace `<pending — see ZENODO.md>` with the concept DOI, and add the badge:
   `[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)`
2. `CITATION.cff` — uncomment and fill the `doi:` field with the concept DOI.
3. `manuscript.md` and `submission/body.tex` — replace the `[AUTHOR]` placeholder in
   *Availability of data and materials* with the **version** DOI.
4. `submission/README.md` — tick the archive item in the pre-submission checklist.
