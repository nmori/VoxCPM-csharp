namespace VoxCPM;

public sealed record GenerationOptions
{
    public string? PromptWavPath { get; init; }
    public string? PromptText { get; init; }
    public string? ReferenceWavPath { get; init; }
    public float CfgValue { get; init; } = 2.0f;
    public int InferenceTimesteps { get; init; } = 10;
    public int MinLen { get; init; } = 2;
    public int MaxLen { get; init; } = 4096;
    public bool Normalize { get; init; }
    public bool Denoise { get; init; }
    public bool RetryBadCase { get; init; } = true;
    public int RetryBadCaseMaxTimes { get; init; } = 3;
    public float RetryBadCaseRatioThreshold { get; init; } = 6.0f;
    public int? Seed { get; init; }
}
