# opencode CLI pinned for the `agentix.agents.opencode` sandbox integration.
# `agentix build` discovers this file through the `agentix.nix` entry point and
# places `opencode` on `/nix/runtime/bin`.
#
# Lifted from numtide/llm-agents.nix (packages/opencode, v1.15.12) and adapted
# to plain nixpkgs: the custom wrapBuddy hook is replaced with the stock
# autoPatchelfHook, and libstdc++ (needed by opencode's bundled @parcel/watcher
# native addon, which is dlopen'd at runtime) is placed on LD_LIBRARY_PATH via
# the wrapper rather than injected as a DT_NEEDED entry.

{ pkgs }:

let
  version = "1.15.12";

  platformMap = {
    "x86_64-linux" = {
      suffix = "linux-x64";
      hash = "sha256-7W+Lzg/qH7K+eJv+DPvnpKgZdw6ZjjKDawcIwBUwOmc=";
    };
    "aarch64-linux" = {
      suffix = "linux-arm64";
      hash = "sha256-7VGBuB6Xj6An2fR81n+Prr7GNCI9f4JKoiYSYYjMvuU=";
    };
  };

  system = pkgs.stdenv.hostPlatform.system;
  platform =
    platformMap.${system} or (throw "agentix-agent-opencode: unsupported system ${system}");

  opencode = pkgs.stdenv.mkDerivation {
    pname = "opencode";
    inherit version;

    src = pkgs.fetchurl {
      url = "https://github.com/anomalyco/opencode/releases/download/v${version}/opencode-${platform.suffix}.tar.gz";
      hash = platform.hash;
    };

    sourceRoot = ".";
    unpackPhase = ''
      runHook preUnpack
      tar -xzf $src
      runHook postUnpack
    '';

    nativeBuildInputs = [
      pkgs.autoPatchelfHook
      pkgs.makeWrapper
    ];
    buildInputs = [ pkgs.stdenv.cc.cc.lib ];

    dontConfigure = true;
    dontBuild = true;
    dontStrip = true; # keep opencode's compressed bun/typescript payload intact

    installPhase = ''
      runHook preInstall
      install -Dm755 opencode $out/bin/opencode
      wrapProgram $out/bin/opencode \
        --prefix PATH : ${pkgs.lib.makeBinPath [ pkgs.fzf pkgs.ripgrep ]} \
        --prefix LD_LIBRARY_PATH : ${pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib ]}
      runHook postInstall
    '';

    meta = {
      description = "opencode — AI coding agent for the terminal";
      homepage = "https://github.com/anomalyco/opencode";
      license = pkgs.lib.licenses.mit;
      mainProgram = "opencode";
      platforms = [ "x86_64-linux" "aarch64-linux" ];
    };
  };
in
pkgs.symlinkJoin {
  name = "agentix-agent-opencode-sys";
  paths = [
    opencode
  ];
}
