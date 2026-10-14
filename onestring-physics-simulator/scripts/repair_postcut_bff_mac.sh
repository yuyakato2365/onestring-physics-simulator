#!/usr/bin/env bash
# Repair official BFF tools inside isolated macOS post-cut worktree.
# Does not change OptCuts or the original OneString worktree.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This repair helper is intended for macOS." >&2
  exit 1
fi
if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew is missing. Install https://brew.sh/ first." >&2
  exit 1
fi

brew install cmake suite-sparse openblas
export CMAKE_PREFIX_PATH="$(brew --prefix suite-sparse);$(brew --prefix openblas)"

# Upstream BFF's FindSuiteSparse.cmake predates current Homebrew layouts.
# It can resolve the libraries while missing SuiteSparse's CHOLMOD headers.
# Give Clang the ACTUAL cholmod.h directory instead of assuming include paths.
suitesparse_prefix="$(brew --prefix suite-sparse)"
cholmod_header="$(find -L "$suitesparse_prefix" -type f -name cholmod.h -print -quit)"
if [[ -z "$cholmod_header" ]]; then
  echo "SuiteSparse is installed but cholmod.h is missing beneath:" >&2
  echo "  $suitesparse_prefix" >&2
  echo "Try: brew reinstall suite-sparse" >&2
  exit 1
fi
cholmod_include="$(dirname "$cholmod_header")"
echo "Using official SuiteSparse CHOLMOD header: $cholmod_header"
source_dir="third_party/boundary-first-flattening"
cli="$source_dir/build/bff-command-line"
rectangle_cli="third_party/bff_rectangle_build/bff-rect-command-line"
existing_bff="$(printenv ONESTRING_BFF_SOURCE 2>/dev/null || true)"

if [[ ! -f "$source_dir/include/bff/project/Bff.h" ]]; then
  if [[ -e "$source_dir" ]]; then
    echo "BFF source directory exists but is incomplete: $source_dir" >&2
    echo "Please inspect or rename it before retrying (no automatic deletion)." >&2
    exit 1
  fi
  mkdir -p third_party
  if [[ -n "$existing_bff" ]]; then
    source_dir="$existing_bff"
    cli="$source_dir/build/bff-command-line"
  else
    git clone --depth 1 --recursive \
      https://github.com/GeometryCollective/boundary-first-flattening.git \
      "$source_dir"
  fi
fi
if [[ ! -f "$source_dir/include/bff/project/Bff.h" ]]; then
  echo "Could not find complete BFF source checkout: $source_dir" >&2
  exit 1
fi

echo "Official BFF source: $source_dir"
echo "Building standard BFF CLI at: $cli"
cmake -S "$source_dir" -B "$source_dir/build" \
  -DBFF_BUILD_GUI=OFF \
  -DBFF_BUILD_CLI=ON \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  "-DCMAKE_CXX_FLAGS=-I$cholmod_include -include cstdint"
cmake --build "$source_dir/build" --target bff-command-line --parallel 4

if [[ ! -x "$cli" ]]; then
  echo "Expected CLI binary is missing after compilation: $cli" >&2
  echo "Check CMake output and look for bff-command-line using:" >&2
  echo "  find '$source_dir/build' -type f -name bff-command-line" >&2
  exit 1
fi

if [[ ! -x "$rectangle_cli" ]]; then
  echo "Rectangular BFF adapter also missing; building it."
  python3 scripts/setup_bff_rectangle.py --source "$source_dir" --jobs 4
fi
if [[ ! -x "$rectangle_cli" ]]; then
  echo "Expected rectangle BFF CLI is missing after compilation: $rectangle_cli" >&2
  exit 1
fi

echo
echo "PASS: both official BFF CLIs are executable:"
ls -lh "$cli" "$rectangle_cli"
echo
echo "Relaunch: python3 -m streamlit run app_postcut_rebff.py"
