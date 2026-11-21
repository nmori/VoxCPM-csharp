using System.Text;

namespace VoxCPM;

public static class WavFile
{
    public static void Write(string path, IReadOnlyList<float> samples, int sampleRate)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        if (sampleRate <= 0)
            throw new ArgumentOutOfRangeException(nameof(sampleRate));
        var directory = Path.GetDirectoryName(Path.GetFullPath(path));
        if (!string.IsNullOrEmpty(directory))
            Directory.CreateDirectory(directory);

        using var stream = File.Create(path);
        using var writer = new BinaryWriter(stream);
        writer.Write("RIFF"u8);
        writer.Write(36 + samples.Count * 2);
        writer.Write("WAVE"u8);
        writer.Write("fmt "u8);
        writer.Write(16);
        writer.Write((short)1);
        writer.Write((short)1);
        writer.Write(sampleRate);
        writer.Write(sampleRate * 2);
        writer.Write((short)2);
        writer.Write((short)16);
        writer.Write("data"u8);
        writer.Write(samples.Count * 2);
        foreach (var sample in samples)
        {
            var clamped = Math.Clamp(sample, -1f, 1f);
            writer.Write((short)Math.Round(clamped * (clamped < 0 ? 32768f : 32767f)));
        }
    }

    public static float[] Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        using var reader = new BinaryReader(File.OpenRead(path));
        var riff = reader.ReadBytes(4);
        _ = reader.ReadInt32();
        var wave = reader.ReadBytes(4);
        if (Encoding.ASCII.GetString(riff) != "RIFF" || Encoding.ASCII.GetString(wave) != "WAVE")
            throw new InvalidDataException("Not a WAVE file.");

        short channels = 1;
        short bitsPerSample = 16;
        int dataSize = 0;
        long dataOffset = 0;
        while (reader.BaseStream.Position < reader.BaseStream.Length)
        {
            var id = reader.ReadBytes(4);
            var size = reader.ReadInt32();
            if (Encoding.ASCII.GetString(id) == "fmt ")
            {
                var format = reader.ReadInt16();
                channels = reader.ReadInt16();
                _ = reader.ReadInt32();
                _ = reader.ReadInt32();
                _ = reader.ReadInt16();
                bitsPerSample = reader.ReadInt16();
                if (size > 16)
                    reader.BaseStream.Position += size - 16;
                if (format != 1)
                    throw new InvalidDataException("Only PCM WAVE files are supported.");
            }
            else if (Encoding.ASCII.GetString(id) == "data")
            {
                dataOffset = reader.BaseStream.Position;
                dataSize = size;
                break;
            }
            else
            {
                reader.BaseStream.Position += size;
            }
        }

        if (dataSize == 0)
            throw new InvalidDataException("WAVE file has no data chunk.");
        if (bitsPerSample != 16)
            throw new InvalidDataException("Only 16-bit PCM WAVE files are supported.");

        reader.BaseStream.Position = dataOffset;
        var frameCount = dataSize / (channels * 2);
        var samples = new float[frameCount];
        for (var i = 0; i < frameCount; i++)
        {
            var sum = 0f;
            for (var channel = 0; channel < channels; channel++)
            {
                var value = reader.ReadInt16();
                sum += value / (value < 0 ? 32768f : 32767f);
            }
            samples[i] = sum / channels;
        }
        return samples;
    }
}
