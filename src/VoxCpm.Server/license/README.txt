voxcpm-tts-server（VoxCPM-csharp）ライセンス一覧
================================================

この配布物に含まれるファイルと、そのライセンスです。各ライセンスの本文はこのフォルダにあります。
（Windows の発行物（win-x64、TorchBackend=cpu）を基準にしています。）

voxcpm-tts-server.exe / voxcpm-tts-server.dll / VoxCpm.dll
    VoxCPM-csharp                        Apache License 2.0   LICENSE.txt, NOTICE.txt
    VoxCPM（OpenBMB, Apache License 2.0, https://github.com/OpenBMB/VoxCPM）の推論部分の C# 移植です。
    一部は Irodori-TTS-csharp（Aratako, MIT License）に由来します（NOTICE.txt に対象と全文があります）。

voices\*.wav / voices\*.txt（男声1〜4・女声1〜4・子供1〜4・英語・韓国語・タイ語・ベトナム語の既定の声）
    VoxCPM2 の声の設計（参照音声なし、説明文とシードは voices\README.txt）で生成した音声と、その読み上げ文です。
    実在の人物の録音は含みません。VoxCPM-csharp と同じ Apache License 2.0 で配布します。

TorchSharp.dll / LibTorchSharp.dll
    TorchSharp 0.107.0 (.NET Foundation) MIT License          TorchSharp_MIT.txt
    https://github.com/dotnet/TorchSharp

libtorch-cpu\*.dll（torch.dll / torch_cpu.dll / torch_global_deps.dll / c10.dll）
    LibTorch (PyTorch) 2.10.0            BSD-3-Clause ほか    LibTorch_PyTorch_BSD-3-Clause.txt
    https://github.com/pytorch/pytorch

libtorch-cpu\libiomp5md.dll / libtorch-cpu\libiompstubs5md.dll
    Intel(R) OpenMP* Runtime Library     Intel End User License Agreement for Developer Tools
                                                              Intel-OpenMP_EULA.txt
                                                              Intel-OpenMP_third-party-programs.txt
    LibTorch に同梱されている再頒布可能ファイルです。下の「利用条件」を参照してください。

libtorch-cpu\uv.dll
    libuv                                MIT License          libuv_MIT.txt

SkiaSharp.dll / libSkiaSharp.dll
    SkiaSharp 2.88.6                     MIT License          SkiaSharp_MIT.txt, SkiaSharp_THIRD-PARTY-NOTICES.txt

Google.Protobuf.dll
    Protocol Buffers 3.21.9              BSD-3-Clause         Google.Protobuf_BSD-3-Clause.txt

ICSharpCode.SharpZipLib.dll
    SharpZipLib 1.4.0                    MIT License          SharpZipLib_MIT.txt

そのほかの .dll（Microsoft.*.dll, System.*.dll, coreclr.dll, hostfxr.dll など）
    .NET 10 / ASP.NET Core ランタイム     MIT License          dotnet_MIT.txt
                                                              dotnet-runtime_THIRD-PARTY-NOTICES.txt
                                                              aspnetcore_THIRD-PARTY-NOTICES.txt


この配布物に含まれず、使うときにダウンロードするもの
    モデルとトークナイザー（openbmb/VoxCPM2。Hugging Face から %LOCALAPPDATA%\VoxCpm\hf または %VOXCPM_CACHE% へ）
        Apache License 2.0（OpenBMB）。https://huggingface.co/openbmb/VoxCPM2
    CUDA 版 LibTorch（NVIDIA の PC で初回に取得。nuget.org の libtorch-cuda-12.8-win-x64）
        PyTorch の BSD-3-Clause と NVIDIA の CUDA / cuDNN の再頒布条件に従います


利用条件
--------

- Intel OpenMP（libiomp5md.dll, libiompstubs5md.dll）について、リバースエンジニアリング・逆コンパイル・
  逆アセンブルを禁止します。また、これらのファイルについて Intel 社およびその供給者は一切の責任を負いません
  （Intel-OpenMP_EULA.txt の 2.1 D (2) および 7 章による条件です）。
- この配布物は無保証です。各ライセンスの免責事項に従います。


音声合成の利用について
----------------------

- 他人の声を本人の同意なく複製・なりすましに使わないでください。
- 生成した音声を、実在の人物の発言であるかのように見せて公開しないでください。
- 詐欺・誹謗中傷など、法令や他人の権利を侵害する目的に使わないでください。
