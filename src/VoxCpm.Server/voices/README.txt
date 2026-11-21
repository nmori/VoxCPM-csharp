VoxCPM2 C# server - voices folder

Put voice files here. The file name (without extension) is the voice id used in "voice" of /v1/audio/speech.
Files can be added while the server runs.

  name.wav               Voice cloning from the clip (a few seconds of clean speech; any sample rate).
  name.wav + name.txt    The exact transcript of name.wav in name.txt (UTF-8): "ultimate cloning",
                         the most faithful reproduction of the voice.
  name.json              Options for the voice, e.g.
                           {"style": "A calm young woman, soft voice", "seed": 42, "cfg_value": 2.0}
                         Without a name.wav it is a designed voice (no reference audio).
                         Next to name.wav it adds options to that clip (style = controllable cloning).
                         It may also name the audio: {"ref_wav": "a.wav", "prompt_wav": "a.wav", "prompt_text": "..."}

  "none"                 Always available (unless --allow-no-ref-voice false): no reference, a new voice each time.

A designed voice changes with every sentence. For a fixed character, synthesize a sample with the designed voice
and save it here as name.wav (with its transcript as name.txt).

Preset voices (bundled)
  The voices below were designed with VoxCPM2 itself (voice "none", the style below, the seed below,
  cfg_value 2.0, 10 timesteps) and saved as name.wav + name.txt, so each keeps one voice. The audio was
  leveled to about -20 dBFS. Delete any you do not need.

  Japanese (text: "こんにちは、はじめまして。今日はとてもいい天気ですね。これから一緒に、いろいろなお話をしましょう。")
    男声1   seed 44  A young man in his twenties, bright and clear voice, friendly and natural tone
    男声2   seed 44  A calm man in his thirties, deep and steady voice, gentle and reassuring
    男声3   seed 44  A middle-aged man in his fifties, low, warm and dignified voice
    男声4   seed 22  An energetic young man, cheerful and lively voice, slightly fast
    女声1   seed 44  A young woman in her twenties, bright, cute and lively voice
    女声2   seed 44  A calm young woman, soft and gentle voice, speaking slowly
    女声3   seed 11  A mature woman in her thirties, clear and composed voice, like a news announcer
    女声4   seed 11  A middle-aged woman in her fifties, warm, kind and relaxed voice
    子供1   seed 11  A little girl about seven years old, high-pitched, cheerful and innocent child voice
    子供2   seed 44  A girl about ten years old, clear and bright child voice, a little shy
    子供3   seed 44  A little boy about seven years old, high-pitched, energetic and playful child voice
    子供4   seed 11  A boy about eleven years old, lively and clear child voice

  Other languages (the text of each is in its name.txt; give these voices text in that language)
    英語・男声        seed 44  An American man in his thirties, clear and friendly voice, native English speaker
    英語・女声        seed 11  An American woman in her twenties, bright and warm voice, native English speaker
    韓国語・男声      seed 33  A Korean man in his twenties, calm and clear voice, native Korean speaker
    韓国語・女声      seed 44  A Korean woman in her twenties, soft and bright voice, native Korean speaker
    タイ語・男声      seed 11  A Thai man in his thirties, gentle and clear voice, native Thai speaker
    タイ語・女声      seed 44  A Thai woman in her twenties, sweet and polite voice, native Thai speaker
    ベトナム語・男声  seed 11  A Vietnamese man in his thirties, warm and clear voice, native Vietnamese speaker
    ベトナム語・女声  seed 33  A Vietnamese woman in her twenties, bright and gentle voice, native Vietnamese speaker
