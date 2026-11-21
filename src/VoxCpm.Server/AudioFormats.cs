// Derived from Irodori-TTS-csharp (MIT License, Copyright (c) 2026 Aratako). See NOTICE.
using VoxCpm.Audio;

namespace VoxCpm.Server;

/// <summary>Response formats of <c>/v1/audio/speech</c>: wav (16-bit PCM), pcm (raw 16-bit little endian), flac.</summary>
public static class AudioFormats
{
    public const string SupportedList = "wav, pcm or flac";

    public static bool IsSupported(string format) => format is "wav" or "pcm" or "flac";

    /// <summary>Formats sent while the audio is generated (a streaming WAV header / raw PCM).</summary>
    public static bool IsProgressive(string format) => format is "wav" or "pcm";

    public static string ContentType(string format) => format switch
    {
        "pcm" => "audio/pcm",
        "flac" => "audio/flac",
        _ => "audio/wav",
    };

    public static byte[] Encode(float[] audio, int sampleRate, string format)
    {
        switch (format)
        {
            case "pcm":
                return WavFile.ToPcm16Bytes(audio);
            case "flac":
                return FlacEncoder.Encode(audio, sampleRate);
            default:
                using (var ms = new MemoryStream())
                {
                    WavFile.WritePcm16(ms, new[] { audio }, sampleRate);
                    return ms.ToArray();
                }
        }
    }
}
