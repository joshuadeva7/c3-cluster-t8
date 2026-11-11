# Make sure the third-party sources the XFeat / LightGlue backends import are present in mac/_src (sourced by the other mac/*.sh),
# then put them on PYTHONPATH (no pip -e install, so nothing points into a build folder that an update deletes).
# Order: already there -> bundled copy shipped in the tgz (mac/_src_bundle) -> git clone at the benchmarked commits.
_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; _SRC="$_HERE/_src"
if [ ! -f "$_SRC/accelerated_features/modules/model.py" ] || [ ! -f "$_SRC/accelerated_features/weights/xfeat.pt" ]; then
  mkdir -p "$_SRC"
  if [ -d "$_HERE/_src_bundle/accelerated_features" ]; then
    echo "[ensure_src] using bundled XFeat sources"; rm -rf "$_SRC/accelerated_features"; cp -R "$_HERE/_src_bundle/accelerated_features" "$_SRC/"
  else
    echo "[ensure_src] cloning XFeat"; rm -rf "$_SRC/accelerated_features"
    git clone -q https://github.com/verlab/accelerated_features.git "$_SRC/accelerated_features" && git -C "$_SRC/accelerated_features" checkout -q e92685f57f8318b18725c5c8c0bd28c7fe188d9a
  fi
fi
if [ ! -f "$_SRC/LightGlue/lightglue/__init__.py" ]; then
  mkdir -p "$_SRC"
  if [ -d "$_HERE/_src_bundle/LightGlue" ]; then rm -rf "$_SRC/LightGlue"; cp -R "$_HERE/_src_bundle/LightGlue" "$_SRC/"
  else git clone -q https://github.com/cvg/LightGlue.git "$_SRC/LightGlue" && git -C "$_SRC/LightGlue" checkout -q 5a650c444718caffc2669dc0094cde908f42dd73; fi
fi
[ -f "$_SRC/accelerated_features/modules/model.py" ] || { echo "[ensure_src] ERROR: XFeat sources missing in $_SRC"; return 1 2>/dev/null || exit 1; }
export LIVE_DEMO_SRC="$_SRC"
export PYTHONPATH="$_SRC/LightGlue:$_SRC/accelerated_features${PYTHONPATH:+:$PYTHONPATH}"
