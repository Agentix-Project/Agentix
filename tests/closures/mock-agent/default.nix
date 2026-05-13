{ pkgs ? import <nixpkgs> {} }:

let
  python = pkgs.python312;
  pythonPkgs = python.pkgs;
in
pythonPkgs.buildPythonApplication {
  pname = "agentix-closure-mock-agent";
  version = "0.1.0";
  format = "pyproject";

  src = ./.;

  nativeBuildInputs = [ pythonPkgs.hatchling ];
  propagatedBuildInputs = [ pythonPkgs.pydantic ];
  doCheck = false;

  # Drop the manifest into $out alongside the Python package, so the closure
  # image's entry symlink resolves /nix/entry/manifest.json correctly.
  postInstall = ''
    cp ${./manifest.json} $out/manifest.json
  '';

  meta.description = "Mock agent closure used in Agentix tests";
}
