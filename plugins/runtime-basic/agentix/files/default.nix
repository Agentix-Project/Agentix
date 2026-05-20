# `agentix.files` is pure-Python (stdlib path ops on /workspace), so it
# brings no system binaries of its own. The file exists for symmetry with
# `agentix.bash` — `agentix build`'s plugin scanner picks up every
# `agentix/<short>/default.nix` it finds, and shipping an explicit empty
# derivation here documents "this namespace deliberately needs nothing"
# instead of relying on absence.

{ pkgs }:

pkgs.symlinkJoin {
  name = "agentix-files-sys";
  paths = [ ];
}
