{
  description = "Skin Lesion Segmentation with UNet";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    poetry2nix.url = "github:nix-community/poetry2nix";
  };

  outputs = { self, nixpkgs, flake-utils, poetry2nix }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [
            (self: super: {
              # CUDA-enabled PyTorch override
              pytorch = super.pytorch-bin.override {
                cudaSupport = true;
                cudaVersion = "12.8";
              };
            })
          ];
        };
      in {
        devShell = pkgs.mkShell {
          buildInputs = with pkgs; [
            python312
            poetry
            gccStdenv
            python312Packages.pylance
            python3Packages.virtualenv
            python312Packages.ipykernel
            jupyter
          ];

          shellHook = ''
            export PYTHONPATH=$PWD:$PYTHONPATH
            export CUDA_VISIBLE_DEVICES=0
            # fixes libstdc++ issues and libgl.so issues
            export LD_LIBRARY_PATH=${pkgs.zlib}/lib:${pkgs.stdenv.cc.cc.lib}/lib:/run/opengl-driver/lib
            # fixes xcb issues :
            QT_PLUGIN_PATH=${pkgs.qt5.qtbase}/${pkgs.qt5.qtbase.qtPluginPrefix}
            poetry install
          '';
        };

      }
    );
}
