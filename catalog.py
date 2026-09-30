"""Model library shown in Settings.

Download sizes come from Hugging Face / the Ollama registry. RAM, speed and quality are rough
CPU-only estimates (int8 for Whisper, Q4 for Ollama) meant for comparing models, not guarantees.
Models marked "recommended" are the light defaults; the first recommended one is used by default.
"""

MULTI, EN = "Multilingual", "English only"


def _w(id, label, mb, ram, speed, quality, langs, good_for, repo=None, recommended=False):
    return {"id": id, "repo": repo or f"Systran/faster-whisper-{id}", "label": label, "bytes": mb * 1e6,
            "ram_gb": ram, "speed": speed, "quality": quality, "langs": langs, "good_for": good_for,
            "recommended": recommended}


def _l(id, label, mb, ram, speed, quality, langs, good_for, recommended=False):
    return {"id": id, "label": label, "bytes": mb * 1e6, "ram_gb": ram, "speed": speed,
            "quality": quality, "langs": langs, "good_for": good_for, "recommended": recommended}


WHISPER_CATALOG = [
    _w("tiny", "Whisper Tiny", 78, 1, "Fastest", "Basic", MULTI,
       "Quick drafts of clear speech and short voice notes. Weak on Hindi/Urdu.", recommended=True),
    _w("base", "Whisper Base", 148, 1, "Very fast", "Fair", MULTI,
       "Everyday recordings with little background noise.", recommended=True),
    _w("small", "Whisper Small", 486, 2, "Moderate", "Good", MULTI,
       "Hindi/Urdu and mixed-language speech, noisy phone or WhatsApp audio.", recommended=True),
    _w("rhasspy/faster-whisper-tiny-int8", "Whisper Tiny · int8", 43, 0.5, "Fastest", "Basic", MULTI,
       "Smallest download of all, for very low-end PCs. Same accuracy as Tiny.",
       repo="rhasspy/faster-whisper-tiny-int8"),
    _w("rhasspy/faster-whisper-base-int8", "Whisper Base · int8", 80, 0.7, "Very fast", "Fair", MULTI,
       "Base accuracy at about half the download.", repo="rhasspy/faster-whisper-base-int8"),
    _w("rhasspy/faster-whisper-small-int8", "Whisper Small · int8", 255, 1.5, "Moderate", "Good", MULTI,
       "Small's accuracy at half the download. A good pick for weak PCs.",
       repo="rhasspy/faster-whisper-small-int8"),
    _w("tiny.en", "Whisper Tiny (English)", 78, 1, "Fastest", "Basic", EN,
       "English-only recordings; a little more accurate than Tiny for English."),
    _w("base.en", "Whisper Base (English)", 148, 1, "Very fast", "Fair", EN,
       "English-only meetings on a low-end PC."),
    _w("small.en", "Whisper Small (English)", 486, 2, "Moderate", "Good", EN,
       "Clear, accurate English transcripts."),
    _w("distil-small.en", "Distil-Whisper Small (English)", 336, 1.5, "Fast", "Good", EN,
       "Faster than Small with similar English accuracy.",
       repo="Systran/faster-distil-whisper-small.en"),
    _w("distil-medium.en", "Distil-Whisper Medium (English)", 792, 2.5, "Moderate", "Very good", EN,
       "High English accuracy at roughly Small's speed.",
       repo="Systran/faster-distil-whisper-medium.en"),
    _w("rhasspy/faster-whisper-medium-int8", "Whisper Medium · int8", 785, 2.5, "Slow", "Very good", MULTI,
       "Medium's accuracy at half the download.", repo="rhasspy/faster-whisper-medium-int8"),
    _w("medium", "Whisper Medium", 1531, 3, "Slow", "Very good", MULTI,
       "Accurate multilingual transcripts when you can wait."),
    _w("medium.en", "Whisper Medium (English)", 1530, 3, "Slow", "Very good", EN,
       "Very accurate English; slow on CPU."),
    _w("Zoont/faster-whisper-large-v3-turbo-int8-ct2", "Whisper Large-v3 Turbo · int8", 818, 3, "Slow",
       "Excellent", MULTI, "Near-best accuracy at half the Turbo download.",
       repo="Zoont/faster-whisper-large-v3-turbo-int8-ct2"),
    _w("large-v3-turbo", "Whisper Large-v3 Turbo", 1622, 4, "Slow on CPU", "Excellent", MULTI,
       "Best accuracy for its speed among the large models. Great for Hindi/Urdu.",
       repo="mobiuslabsgmbh/faster-whisper-large-v3-turbo"),
    _w("deepdml/faster-whisper-large-v3-turbo-ct2", "Whisper Large-v3 Turbo (deepdml)", 1622, 4,
       "Slow on CPU", "Excellent", MULTI, "Alternative build of Large-v3 Turbo.",
       repo="deepdml/faster-whisper-large-v3-turbo-ct2"),
    _w("distil-large-v2", "Distil-Whisper Large-v2 (English)", 1516, 4, "Slow", "Excellent", EN,
       "Distilled large model for English audio.", repo="Systran/faster-distil-whisper-large-v2"),
    _w("distil-large-v3", "Distil-Whisper Large-v3 (English)", 1516, 4, "Slow", "Excellent", EN,
       "Distilled large model for English audio.", repo="Systran/faster-distil-whisper-large-v3"),
    _w("distil-large-v3.5", "Distil-Whisper Large-v3.5 (English)", 1516, 4, "Slow", "Excellent", EN,
       "Newest distilled large model, English only.", repo="distil-whisper/distil-large-v3.5-ct2"),
    _w("large-v1", "Whisper Large-v1", 3090, 6, "Very slow", "Very good", MULTI,
       "Original large model; mostly useful for comparison."),
    _w("large-v2", "Whisper Large-v2", 3090, 6, "Very slow", "Excellent", MULTI,
       "Strong multilingual accuracy; very slow on CPU."),
    _w("large-v3", "Whisper Large-v3", 3091, 6, "Very slow", "Best", MULTI,
       "Highest accuracy; realistically needs a GPU."),
]

LLM_CATALOG = [
    _l("smollm2:135m", "SmolLM2 · 135M", 271, 0.5, "Fastest", "Very basic", EN,
       "Tiny test model; minutes will be rough. The transcript must already be English."),
    _l("qwen2.5:0.5b", "Qwen 2.5 · 0.5B", 398, 1, "Fastest", "Basic", MULTI,
       "Short voice notes and a quick bullet summary. May miss details or mistranslate.",
       recommended=True),
    _l("qwen3:0.6b", "Qwen 3 · 0.6B", 523, 1, "Fastest", "Basic", MULTI,
       "Newer tiny Qwen; slightly better structure than Qwen 2.5 0.5B."),
    _l("tinyllama:1.1b", "TinyLlama · 1.1B", 638, 1.5, "Very fast", "Basic", EN,
       "Older small model; English transcripts only."),
    _l("gemma3:1b", "Gemma 3 · 1B", 815, 1.5, "Very fast", "Fair", EN,
       "Short English meetings."),
    _l("qwen2.5:1.5b", "Qwen 2.5 · 1.5B", 986, 2, "Fast", "Good", MULTI,
       "Most meetings under ~30 min, with decent English from Hindi/Urdu.", recommended=True),
    _l("deepseek-r1:1.5b", "DeepSeek-R1 · 1.5B", 1117, 2, "Slow (reasons first)", "Fair", MULTI,
       "Reasoning model: it thinks before writing, so it's slower. Better at logic than summaries."),
    _l("llama3.2:1b", "Llama 3.2 · 1B", 1321, 2, "Very fast", "Fair", MULTI,
       "Short meetings; Hindi is among its supported languages."),
    _l("qwen3:1.7b", "Qwen 3 · 1.7B", 1359, 2.5, "Fast", "Good", MULTI,
       "Strong small model with good translation into English."),
    _l("granite3.3:2b", "Granite 3.3 · 2B", 1545, 2.5, "Fast", "Good", MULTI,
       "IBM's business-focused model; tidy, structured minutes."),
    _l("gemma2:2b", "Gemma 2 · 2B", 1630, 2.5, "Fast", "Good", MULTI,
       "Well-written English summaries of short meetings."),
    _l("smollm2:1.7b", "SmolLM2 · 1.7B", 1820, 2.5, "Fast", "Fair", EN,
       "Compact English summaries."),
    _l("qwen2.5:3b", "Qwen 2.5 · 3B", 1930, 4, "Moderate", "Very good", MULTI,
       "Longer meetings, clearer action items and better translation.", recommended=True),
    _l("llama3.2:3b", "Llama 3.2 · 3B", 2019, 4, "Moderate", "Good", MULTI,
       "Reliable summaries and action items; a good all-rounder."),
    _l("phi3.5:3.8b", "Phi-3.5 Mini · 3.8B", 2176, 4.5, "Moderate", "Good", MULTI,
       "Good reasoning and a long context window for long meetings."),
    _l("phi4-mini:3.8b", "Phi-4 Mini · 3.8B", 2492, 4.5, "Moderate", "Very good", MULTI,
       "Strong instruction following; well-structured minutes."),
    _l("qwen3:4b", "Qwen 3 · 4B", 2497, 4.5, "Moderate", "Very good", MULTI,
       "Best quality under 3 GB; great Hindi/Urdu to English."),
    _l("gemma3:4b", "Gemma 3 · 4B", 3339, 5, "Moderate", "Very good", MULTI,
       "Excellent multilingual understanding and polished English."),
    _l("mistral:7b", "Mistral · 7B", 4373, 6, "Slow", "Good", "Mostly English / European",
       "Solid English summaries; weaker on Hindi/Urdu."),
    _l("qwen2.5:7b", "Qwen 2.5 · 7B", 4683, 6.5, "Slow", "Excellent", MULTI,
       "High-quality minutes and translation; needs ~7 GB of free RAM."),
    _l("llama3.1:8b", "Llama 3.1 · 8B", 4921, 7, "Slow", "Excellent", MULTI,
       "Detailed, well-organised minutes for long meetings."),
    _l("aya-expanse:8b", "Aya Expanse · 8B", 5057, 7, "Slow", "Excellent", MULTI,
       "Built for multilingual work; excellent Hindi to English."),
    _l("qwen3:8b", "Qwen 3 · 8B", 5225, 7, "Slow", "Excellent", MULTI,
       "Top-tier small-model quality; heavy for CPU-only systems."),
    _l("gemma2:9b", "Gemma 2 · 9B", 5443, 7.5, "Very slow", "Excellent", MULTI,
       "Very polished writing; slow without a GPU."),
    _l("gemma3:12b", "Gemma 3 · 12B", 8149, 10, "Very slow", "Best", MULTI,
       "Best quality here, but too heavy for most 8–12 GB RAM systems."),
]

WHISPER_BY_ID = {m["id"]: m for m in WHISPER_CATALOG}
LLM_BY_ID = {m["id"]: m for m in LLM_CATALOG}
DEFAULT_WHISPER = next(m["id"] for m in WHISPER_CATALOG if m["recommended"])
DEFAULT_LLM = next(m["id"] for m in LLM_CATALOG if m["recommended"])
