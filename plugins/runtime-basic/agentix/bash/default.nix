# System binaries the `agentix.bash` namespace expects on PATH inside
# the sandbox. `agentix build` discovers this file via
# `importlib.resources.files('agentix.bash') / 'default.nix'` after
# `pip install agentix-runtime-basic` lands it next to __init__.py.
#
# The function form `{ pkgs }: drv` is the plugin Nix convention: the
# builder hands every plugin the same Nixpkgs, so all plugins share one
# revision (no per-plugin version drift).

{ pkgs }:

pkgs.symlinkJoin {
  name = "agentix-bash-sys";
  paths = with pkgs; [
    bashInteractive
    coreutils
  ];
}
