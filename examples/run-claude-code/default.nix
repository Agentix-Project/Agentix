{ pkgs }:

pkgs.symlinkJoin {
  name = "run-claude-code-sys";
  paths = with pkgs; [
    git
  ];
}
