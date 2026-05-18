{
  description = "Agentix bundle builder — turns a user's pyproject + uv.lock + plugin default.nix files into a self-contained runtime image";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-25.05";

    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, pyproject-nix, uv2nix, pyproject-build-systems }:
    let
      forAllSystems = nixpkgs.lib.genAttrs [ "x86_64-linux" "aarch64-linux" ];
    in
    {
      # The library entry point `agentix build` calls into.
      #
      #   mkBundle {
      #     pkgs           = nixpkgs.legacyPackages.${system};
      #     name           = "hello-agentix";
      #     tag            = "0.1.0";
      #     workspaceRoot  = ./project;     # pyproject.toml + uv.lock live here
      #     pluginNixFiles = [ ./agentix-bash.nix ./user-project.nix ];
      #     pythonVersion  = "311";         # picks pkgs.python<version>
      #   }
      #
      # Result: a stream-layered docker image. `docker load < $result` loads it.
      lib = {
        mkBundle = import ./builder.nix {
          inherit pyproject-nix uv2nix pyproject-build-systems;
        };
      };

      # Smoke-test target — verifies the toolchain on a stub project.
      # Build with `nix build .#poc`; load with `./result | docker load`.
      packages = forAllSystems (system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          python = pkgs.python311;
          pythonEnv = python.withPackages (ps: with ps; [ fastapi pydantic httpx ]);
        in
        {
          poc = pkgs.dockerTools.streamLayeredImage {
            name = "agentix-poc";
            tag = "latest";
            contents = [ pythonEnv ];
            config = {
              Entrypoint = [
                "${pythonEnv}/bin/python"
                "-c"
                "import fastapi, pydantic, httpx; print('agentix nix poc OK:', fastapi.__version__, pydantic.VERSION, httpx.__version__)"
              ];
              Env = [ "PATH=${pythonEnv}/bin" ];
            };
          };
        });
    };
}
