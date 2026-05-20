{
  description = "agentix bundle builder — Nix toolchain + system-dep closures";

  # Only nixpkgs. The Python side is uv's job (uv venv + uv sync inside
  # the build container), so there is no uv2nix / pyproject.nix here —
  # Nix never touches Python packaging.
  inputs.nixpkgs.url = "github:nixos/nixpkgs/nixos-25.05";

  outputs =
    { self, nixpkgs }:
    let
      inherit (nixpkgs) lib;
      system = lib.removeSuffix "\n" (builtins.readFile ./nix-system);

      pkgs = import nixpkgs {
        inherit system;
        # Plugin closures may pull unfree binaries (e.g. the claude CLI).
        config.allowUnfree = true;
      };

      # Interpreter minor (e.g. "311"), written by the host into
      # `python-version` so this flake ships verbatim — no templating.
      pythonMinor = lib.removeSuffix "\n" (builtins.readFile ./python-version);
      python = pkgs."python${pythonMinor}";

      # System-dep closures the in-container `_assemble` step stages
      # into `closures/` — one `{ pkgs }: drv` per plugin/project. The
      # directory is empty when `toolchain` is built (before assembly),
      # so guard on existence.
      closuresDir = ./closures;
      closureFiles =
        if builtins.pathExists closuresDir then
          lib.filter (lib.hasSuffix ".nix") (builtins.attrNames (builtins.readDir closuresDir))
        else
          [ ];
      closureDrvs = map (f: import (closuresDir + "/${f}") { inherit pkgs; }) closureFiles;
    in
    {
      packages.${system} = {
        # Built first — the interpreter + uv. The container uses these
        # to create `/nix/runtime/venv` and run `uv sync`.
        toolchain = pkgs.symlinkJoin {
          name = "agentix-toolchain";
          paths = [
            python
            pkgs.uv
          ];
        };

        # Built last — toolchain plus every discovered system-dep
        # closure, merged into one tree of symlinks into /nix/store.
        # This tree is copied to /nix/runtime in the image.
        runtime = pkgs.symlinkJoin {
          name = "agentix-runtime";
          paths = [
            python
            pkgs.uv
          ]
          ++ closureDrvs;
        };
      };
    };
}
