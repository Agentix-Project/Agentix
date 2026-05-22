# Claude Code CLI pinned for the `agentix.agents.claude_code` sandbox
# integration. `agentix build` discovers this file through the
# `agentix.nix` entry point and places `claude` on `/nix/runtime/bin`.

{ pkgs }:

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
  name = "agentix-agent-claude-code-sys";
  paths = [
    claude
  ];
}
