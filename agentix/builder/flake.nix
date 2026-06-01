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

      # The project's own [tool.agentix].nix closure, staged by the
      # in-container `_assemble` step as `closures/project.nix`. It
      # merges into the `runtime` tree alongside the toolchain —
      # these are the binaries the user's *own* code calls. Optional;
      # the directory is empty for pure-Python projects.
      projectFile = ./closures/project.nix;
      projectPaths =
        if builtins.pathExists projectFile then
          [ (import projectFile { inherit pkgs; }) ]
        else
          [ ];

      # Plugin closures, staged as `closures/plugins/<label>.nix`.
      # Each lands at `/nix/runtime/plugins/<label>/` as its own
      # `/nix/store` tree — *no merge*. bootstrap.sh assembles
      # PATH / LD_LIBRARY_PATH / ... by globbing this dir, so two
      # plugins legitimately shipping the same binary (`pkgs.git`,
      # `bash` vs `bashInteractive`, ...) can never collide at build
      # time. First-wins is decided by PATH lookup order, which is
      # bootstrap's job to make deterministic. Same mental model as
      # `nix-shell -p a b c`.
      pluginsDir = ./closures/plugins;
      pluginEntries =
        if builtins.pathExists pluginsDir then
          map
            (f: {
              name = lib.removeSuffix ".nix" f;
              path = import (pluginsDir + "/${f}") { inherit pkgs; };
            })
            (lib.filter (lib.hasSuffix ".nix") (builtins.attrNames (builtins.readDir pluginsDir)))
        else
          [ ];
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

        # Toolchain + project closure, merged into one tree of symlinks
        # into /nix/store. Copied to /nix/runtime/{bin,lib,...} in the
        # image. `symlinkJoin` (not buildEnv): the project plus toolchain
        # is a small, known set; a real collision here (e.g., a project
        # closure trying to bring a different python) is a bug worth
        # surfacing loudly, not a soft warning.
        #
        # `pkgs.bash` is part of every bundle so `bootstrap.sh` can rely
        # on a known-good shell at `/nix/runtime/bin/bash` regardless of
        # what `/bin/sh` the task image ships. Lets bootstrap.sh use
        # bash-specific features (`set -o pipefail`, parameter-expansion
        # forms, etc.) without sniffing the task image first.
        runtime = pkgs.symlinkJoin {
          name = "agentix-runtime";
          paths = [
            python
            pkgs.uv
            pkgs.bash
          ]
          ++ projectPaths;
        };

        # `/nix/runtime/plugins/<label> -> /nix/store/<plugin>`. Each
        # plugin's `bin/`, `lib/`, ... stays self-contained. The whole
        # dir is exposed as one `linkFarm` so bundle-build.sh does a
        # single `nix build .#plugins` regardless of how many plugins
        # registered. Empty (= empty dir) when there are no plugins.
        plugins = pkgs.linkFarm "agentix-plugins" pluginEntries;
      };
    };
}
