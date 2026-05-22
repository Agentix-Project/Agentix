# System binaries and shared libraries the `agentix.plugins.datasets.swe`
# evaluator expects inside SWE-bench task images.
#
# The function form `{ pkgs }: drv` is the plugin Nix convention: the
# builder hands every plugin the same Nixpkgs revision and merges the
# returned derivations into `/nix/runtime`.

{ pkgs }:

pkgs.symlinkJoin {
  name = "agentix-dataset-swe-sys";
  paths = with pkgs; [
    git
    patch
    # Binary wheels used by the SWE-bench harness stack (numpy, pandas,
    # pyarrow, scikit-learn) need libstdc++ at runtime.
    stdenv.cc.cc.lib
  ];
}
