{
  description = "TinyTAuK development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" ];
      forAllSystems = nixpkgs.lib.genAttrs systems;
    in {
      devShells = forAllSystems (system:
        let
          pkgs = import nixpkgs { inherit system; };
        in {
          default = pkgs.mkShell {
            packages = with pkgs; [
              python313
              uv
              ruff
              ffmpeg
              libsndfile
              zlib
              pkg-config
              git
              gcc
            ];

            shellHook = ''
              export UV_PROJECT_ENVIRONMENT="$PWD/.venv"
              export UV_PYTHON="${pkgs.python313}/bin/python3"
              export UV_NO_MANAGED_PYTHON=1
              export PYTHONUNBUFFERED=1
              export LD_LIBRARY_PATH="${pkgs.ffmpeg.lib}/lib:${pkgs.libsndfile}/lib:${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.zlib}/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

              echo "TinyTAuK dev shell: Python $(python --version 2>&1), uv $(uv --version 2>&1)"
              echo "Run: uv sync && ./scripts/check"
            '';
          };
        });
    };
}
