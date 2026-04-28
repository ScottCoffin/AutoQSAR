# AutoQSAR
An auto-QSAR workflow designed for learning and full-scale operation

## Installation
### Pre-requisites
The following are necessary to run the core workflow of this repository:
* A computer with >4GB RAM (>32 GB RAM recommended for )

### Environment
It is highly recommended to install the Conda environment. First, Conda should be installed via their wesbite. Miniconda is recommended over Anaconda to enable faster installation and conserve space on your machine.
If this is your first time installing Miniconda or Anaconda, following installation you must open the Anaconda Powershell and run `conda init powershell` to enable the use of conda in a powershell terminal in your IDE. You may also need to open a Powershell as Administrator and run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser.`

Once your conda installation is complete, the easiest way to set up the environment is to run the provided setup script, which handles both steps automatically:

```bat
setup_env.bat
```

Or on PowerShell:

```powershell
.\setup_env.ps1
```

Both scripts will prompt you to choose CUDA (GPU), CPU-only, or custom (manual PyTorch) and then complete the full installation.

**Manual installation** (if you prefer step-by-step):

The setup is a two-step process. First, conda creates the base environment, then `uv` installs all pip packages. `uv` is used instead of pip directly because the ~300 pip dependencies are too complex for pip's resolver to handle in a single pass.

**A note on `--index-strategy unsafe-best-match`**

The CUDA install uses two package indexes: [PyPI](https://pypi.org) (the standard index) and the [PyTorch CUDA wheel index](https://download.pytorch.org/whl/cu121) (for GPU-enabled builds of torch, torchvision, and torchaudio). By default, uv applies a "first index wins" rule: once it finds a package name on any index, it only considers versions from that index. This is a deliberate security measure called *index priority*, designed to prevent [dependency confusion attacks](https://medium.com/@alex.birsan/dependency-confusion-4a5d60fec610) where a malicious package on a public index shadows a legitimate private one.

The problem is that the PyTorch index also incidentally hosts a handful of common packages (e.g., `certifi`) at outdated versions. Under the default strategy, uv finds `certifi` on the PyTorch index first and refuses to install the newer PyPI version, making the environment unsolvable.

`--index-strategy unsafe-best-match` lifts the "first index wins" restriction and instead picks the best-matching version of each package across all configured indexes — which is how pip has always behaved. The word "unsafe" in the flag name reflects the theoretical dependency confusion risk, not a risk specific to this project.

**In practice the risk here is negligible:** both indexes used (`pypi.org` and `download.pytorch.org`) are operated by trusted organisations (PSF and Meta), the package list is fully pinned to exact versions, and uv will still reject any package whose version does not exactly match what is specified in `requirements-cuda.txt`.

With CUDA (requires NVIDIA GPU with CUDA 12.1+):
```bat
conda env create -f environment-cuda.yml -n autoqsar-py311
conda run -n autoqsar-py311 pip install uv
conda run -n autoqsar-py311 uv pip install --system --index-strategy unsafe-best-match --prerelease=allow -r requirements-cuda.txt
```

CPU-only:
```bat
conda env create -f environment-cpu.yml -n autoqsar-py311
conda run -n autoqsar-py311 pip install uv
conda run -n autoqsar-py311 uv pip install --system --index-strategy unsafe-best-match --prerelease=allow -r requirements-cpu.txt
```

This may take 15 to 30 minutes depending on your machine and internet connection. Once complete, activate the environment with:

```bat
conda activate autoqsar-py311
```