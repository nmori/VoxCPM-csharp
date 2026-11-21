"""Golden token ids of the VoxCPM2 text tokenizer (LlamaTokenizerFast + mask_multichar_chinese_tokens)."""
import json
import os

import _common  # noqa: F401
from transformers import LlamaTokenizerFast
from voxcpm.model.utils import mask_multichar_chinese_tokens

CASES = [
    "Hello, world!",
    "VoxCPM2 is the current recommended release for realistic multilingual speech synthesis.",
    "こんにちは、私はAIです。これは音声合成のテストです。",
    "今日はいい天気ですね。東京都渋谷区に行きました。",
    "你好世界，我们今天去北京。",
    "(A young woman, gentle and sweet voice)Hello, welcome to VoxCPM2!",
    "(年轻女性，温柔甜美)欢迎使用语音合成。",
    "  leading and   multiple   spaces  ",
    "Numbers 12345 and 3.14, dates 2026-11-21.",
    "絵文字🤭😮‍💨を含む文章。",
    "한국어 문장입니다. Ceci est une phrase française. Это русское предложение.",
    "ภาษาไทย ကမ္ဘာ ສະບາຍດີ",
    "Tabs\tand\nnewlines",
    "<|im_start|>special tokens<|im_end|>",
    "<|background|>rain<|/background|>hi <|audio_start|>x",
    "<s>a</s><unk> <|audio_prompt_start|><|ref_audio_start|>",
    "Ünïcödé àccents — dashes … ellipsis “quotes”",
    "a",
    "粤语：你食咗饭未呀？",
    "龘靐齉𠀀",
]


def main():
    tok = LlamaTokenizerFast.from_pretrained(_common.CKPT)
    wrapped = mask_multichar_chinese_tokens(tok)
    out = [{"text": t, "ids": [int(i) for i in wrapped(t)], "base_ids": tok(t, add_special_tokens=False)["input_ids"]} for t in CASES]
    os.makedirs(_common.DATA_DIR, exist_ok=True)
    path = os.path.join(_common.DATA_DIR, "text_ref.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("wrote", path, sum(len(c["ids"]) for c in out), "ids")


if __name__ == "__main__":
    main()
