{ pkgs ? import <nixpkgs> {} }:

let
  python = pkgs.python312;
  pythonPkgs = python.pkgs;
in
pythonPkgs.buildPythonApplication {
  pname = "agentix-closure-mock-dataset";
  version = "0.1.0";
  format = "pyproject";

  src = ./.;

  nativeBuildInputs = [ pythonPkgs.hatchling ];
  propagatedBuildInputs = [ pythonPkgs.pydantic ];
  doCheck = false;

  postInstall = ''
    cp ${./manifest.json} $out/manifest.json
  '';

  meta.description = "Mock dataset closure used in Agentix tests";
}
