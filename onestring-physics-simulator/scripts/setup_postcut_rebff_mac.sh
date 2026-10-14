#!/usr/bin/env bash
# One-time macOS prerequisites for the isolated OneString Split -> re-BFF experiment.
# Keeps official BFF source and two binary builds separate from the main pipeline.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This setup helper is for macOS." >&2
  exit 1
fi
if ! command -v brew >/dev/null 2>&1; then
  echo "Install Homebrew (https://brew.sh/) before continuing." >&2
  exit 1
fi

brew install cmake suite-sparse openblas python@3.11
export CMAKE_PREFIX_PATH="$(brew --prefix suite-sparse);$(brew --prefix openblas)"
if ! command -v python3.11 >/dev/null 2>&1; then
  echo "Could not find python3.11 after Homebrew installation." >&2
  exit 1
fi

python3.11 -m venv .venv-postcut-rebff
source .venv-postcut-rebff/bin/activate
python -m pip install -e .

bff_src="third_party/boundary-first-flattening"
if [[ ! -f "$bff_src/include/bff/project/Bff.h" ]]; then
  if [[ -e "$bff_src" ]]; then
    echo "Existing $bff_src is incomplete. Please choose a separate BFF source directory." >&2
    exit 1
  fi
  mkdir -p third_party
  git clone --depth 1 --recursive https://github.com/GeometryCollective/boundary-first-flattening.git "$bff_src"
fi

cmake -S "$bff_src" -B "$bff_src/build" \
  -DBFF_BUILD_GUI=OFF -DBFF_BUILD_CLI=ON \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build "$bff_src/build" --target bff-command-line --parallel 4
python scripts/setup_bff_rectangle.py --source "$bff_src" --jobs 4

mkdir -p "$HOME/Downloads"
if [[ ! -f "$HOME/Downloads/lilium.obj" ]]; then
  curl -fL https://raw.githubusercontent.com/libigl/libigl-tutorial-data/master/lilium.obj \
    -o "$HOME/Downloads/lilium.obj"
fi

echo
echo "Setup complete. Run:"
echo "  cd $(pwd)"
echo "  source .venv-postcut-rebff/bin/activate"
echo "  python -m streamlit run app_postcut_rebff.py"
