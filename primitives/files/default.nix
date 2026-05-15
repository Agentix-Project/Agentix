{ pkgs ? import <nixpkgs> {} }:

let
  python = pkgs.python312;
  pythonPkgs = python.pkgs;
in
pythonPkgs.buildPythonApplication {
  pname = "agentix-primitive-files";
  version = "0.1.0";
  format = "pyproject";

  src = ./.;

  nativeBuildInputs = [ pythonPkgs.hatchling ];
  propagatedBuildInputs = [];
  doCheck = false;

  # manifest.json is derived from the package's `__init__.py` at build time
  # (see `tools/gen_manifest.py`).
  postInstall = ''
    ${python}/bin/python ${../../tools/gen_manifest.py} \
      --init "$out/${python.sitePackages}/agentix_closures/files/__init__.py" \
      --out  "$out/manifest.json"
  '';

  meta.description = "Sandbox file I/O primitive — upload / download";
}
