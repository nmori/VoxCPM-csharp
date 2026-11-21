using VoxCPM;

if (args.Length < 4)
{
    Console.Error.WriteLine("Usage: VoxCPM.Cli --model <path> --text <text> --output <file.wav>");
    return 1;
}

string? modelPath = null;
string? text = null;
string? outputPath = null;
string? promptWav = null;
string? promptText = null;
string? referenceWav = null;

for (var i = 0; i < args.Length; i++)
{
    switch (args[i])
    {
        case "--model": modelPath = args[++i]; break;
        case "--text": text = args[++i]; break;
        case "--output": outputPath = args[++i]; break;
        case "--prompt-audio": promptWav = args[++i]; break;
        case "--prompt-text": promptText = args[++i]; break;
        case "--reference-audio": referenceWav = args[++i]; break;
        default: Console.Error.WriteLine($"Unknown argument: {args[i]}"); return 1;
    }
}

if (modelPath is null || text is null || outputPath is null)
{
    Console.Error.WriteLine("Missing required --model, --text, or --output argument.");
    return 1;
}

try
{
    using var model = VoxCPMModel.FromPretrained(modelPath);
    var samples = model.Generate(text, new GenerationOptions
    {
        PromptWavPath = promptWav,
        PromptText = promptText,
        ReferenceWavPath = referenceWav
    });
    WavFile.Write(outputPath, samples, model.SampleRate);
    Console.WriteLine(outputPath);
    return 0;
}
catch (Exception exception)
{
    Console.Error.WriteLine(exception.Message);
    return 1;
}
