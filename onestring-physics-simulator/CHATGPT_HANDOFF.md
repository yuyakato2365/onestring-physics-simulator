# OneString ChatGPT handoff

This repository should be treated as the authoritative OneString state from
Yuya's local work PC.

As of this handoff, the known-good local base is:

```text
03a92394deda9063cebf3adf2f38e011c2ed6983
Stabilize OneString Omega and hinge pipeline
```

The GitHub `main` branch was later advanced on another PC to commits after
`03a9239`. Those newer remote commits were reported to be much slower and
should not be treated as the trusted baseline unless Yuya explicitly asks to
reinvestigate them.

## What matters most

- Keep the current work-PC pipeline behavior as the baseline.
- Do not silently merge later remote changes into `app.py` or
  `src/onestring_physics/onestring_pipeline.py`.
- Recreate the same Python/CUDA environment before judging performance.
- Use the documents in `docs/home_pc_codex_handoff_ja.md` and
  `docs/current_algorithm_overview_ja.md` before making performance changes.

## Reproduction environment

The work-PC environment that was inspected on 2026-07-09 was:

```text
OS: Windows 11
Python: 3.12.13
GPU: NVIDIA GeForce RTX 4080 Laptop GPU
NVIDIA driver: 581.95
Driver-reported CUDA: 13.0
PyTorch: 2.11.0+cu128
PyTorch CUDA runtime: 12.8
```

Use:

```powershell
cd C:\path\to\onestring-physics-simulator
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-local-cu128-lock.txt
pip install -e .
```

If the lock file cannot be installed on the target machine, install the CUDA
PyTorch wheel first:

```powershell
pip install -r requirements-gpu-cu128.txt
pip install -r requirements.txt
pip install -e ".[dev]"
```

Then verify:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
python -m pytest tests -q
streamlit run app.py
```

## Slowdown suspects

If the home PC remains slower after using the same environment, inspect these
before changing the optimizer:

- Whether Streamlit is using the intended `.venv` Python executable.
- Whether `torch.cuda.is_available()` is true inside the Streamlit process.
- Whether the app reports the CUDA backend only for stages that actually stay
  on device.
- Whether `S -> Omega -> M2D` creates too many tiles or hinge candidates.
- Whether `m2d_general_omega_overlay_rebuilt` is true in a case that should use
  the rectangular/default path.
- Whether `fast_t2d` or later remote UI/pipeline edits reintroduced a heavy
  T2D/Dual Hinge path.

The default code path should keep the aggressive post-`03a9239` remote changes
out unless they are deliberately reviewed and re-applied.
