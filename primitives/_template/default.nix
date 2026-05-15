{ pkgs ? import <nixpkgs> {} }:

# Shared nix derivation for every primitive closure.
#
# `tools/build_closure.py` stages a build context with a generated
# `pyproject.toml` (pname/version filled in) and a copy of
# `gen_manifest.py`. This file is the same for every closure — the
# closure-specific bits are pulled from `pyproject.toml` by hatchling
# and from `__init__.py` by `gen_manifest`.

let
  python = pkgs.python312;
  pythonPkgs = python.pkgs;
  pyproject = builtins.fromTOML (builtins.readFile ./pyproject.toml);
  pname = pyproject.project.name;
  version = pyproject.project.version;
  # `pyproject.toml` lists the in-package path; derive the import path
  # for `gen_manifest`'s `--init` argument from it.
  pkgPath = builtins.elemAt pyproject.tool.hatch.build.targets.wheel.packages 0;
in
pythonPkgs.buildPythonApplication {
  inherit pname version;
  format = "pyproject";
  src = ./.;

  nativeBuildInputs = [ pythonPkgs.hatchling ];
  propagatedBuildInputs = [];
  doCheck = false;

  postInstall = ''
    ${python}/bin/python ${./gen_manifest.py} \
      --init "$out/${python.sitePackages}/${pkgPath}/__init__.py" \
      --out  "$out/manifest.json"
  '';

  meta.description = "Agentix closure (${pname})";
}
