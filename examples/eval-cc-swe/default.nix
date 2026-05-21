# System binaries for the eval-cc-swe bundle.
#
# `agentix build` runs this derivation in a builder stage and symlinks
# the resulting `bin/*` into `/nix/runtime/bin/` inside the bundle
# image. Every worker subprocess inherits `/nix/runtime/bin/` on PATH,
# so user code can `subprocess.run("claude", ...)` or call `git` by
# bare name.
#
# Only system binaries belong here. The Python package itself is
# installed into `/nix/runtime/`'s venv by `pip install`, not by Nix.
{ pkgs ? import <nixpkgs> { config.allowUnfree = true; } }:

let
  claude = pkgs.stdenv.mkDerivation (finalAttrs: {
    pname = "claude-code";
    version = "2.1.114";

    src = pkgs.fetchurl {
      url = "https://storage.googleapis.com/claude-code-dist-86c565f3-f756-42ad-8dfa-d59b1c096819/claude-code-releases/${finalAttrs.version}/linux-x64/claude";
      hash = "sha256-Er1LCRbesGvhf/x7LwSF4UC/ALLbPct4Rp1mcj1zwn8=";
    };

    dontUnpack = true;
    dontStrip = true;

    nativeBuildInputs = [ pkgs.makeWrapper ];

    installPhase = ''
      runHook preInstall
      install -Dm755 $src $out/bin/claude
      runHook postInstall
    '';

    postFixup = ''
      wrapProgram $out/bin/claude \
        --argv0 claude \
        --set DISABLE_AUTOUPDATER 1 \
        --set-default DISABLE_NON_ESSENTIAL_MODEL_CALLS 1 \
        --set DISABLE_INSTALLATION_CHECKS 1
    '';

    meta = {
      description = "Anthropic Claude Code CLI";
      homepage = "https://claude.ai/code";
      license = pkgs.lib.licenses.unfree;
      mainProgram = "claude";
      platforms = [ "x86_64-linux" ];
    };
  });
in
pkgs.symlinkJoin {
  name = "eval-cc-swe-deps";
  paths = [
    claude
    pkgs.git
    # Binary wheels used by the SWE-bench harness stack (numpy, pandas,
    # pyarrow, scikit-learn) need libstdc++ at runtime.
    pkgs.stdenv.cc.cc.lib
  ];
}
