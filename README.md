# VoxCPM-csharp

A C# API and command-line wrapper for the [VoxCPM](https://github.com/OpenBMB/VoxCPM) text-to-speech system.

## Requirements

- .NET 8 SDK or later
- A downloaded VoxCPM model snapshot containing `config.json`
- A C# implementation of `IVoxCpmRuntime` that provides the native model inference

The runtime interface allows this repository to use an ONNX, GGUF, native llama.cpp, or platform-specific backend without coupling the public API to a particular inference engine.

## Library usage

```csharp
using VoxCPM;

using var model = VoxCPMModel.FromPretrained(
    "/path/to/VoxCPM",
    runtime: myRuntime);

float[] audio = model.Generate(
    "Hello from VoxCPM.",
    new GenerationOptions
    {
        CfgValue = 2.0f,
        InferenceTimesteps = 10,
        Seed = 42
    });

WavFile.Write("demo.wav", audio, model.SampleRate);
```

Voice cloning options match the Python API:

```csharp
var audio = model.Generate("This is a cloned voice.", new GenerationOptions
{
    ReferenceWavPath = "voice.wav"
});
```

Ultimate cloning can provide both continuation audio and its exact transcript:

```csharp
var audio = model.Generate("This is an ultimate cloning demonstration.", new GenerationOptions
{
    PromptWavPath = "voice.wav",
    PromptText = "The transcript of the reference audio.",
    ReferenceWavPath = "voice.wav"
});
```

## CLI

```bash
dotnet run --project src/VoxCPM.Cli -- \
  --model /path/to/VoxCPM \
  --text "Hello world" \
  --output out.wav
```

Optional cloning arguments:

```text
--prompt-audio <wav> --prompt-text <text>
--reference-audio <wav>
```

## Build and test

```bash
dotnet build VoxCPM.slnx
dotnet test VoxCPM.slnx
```
