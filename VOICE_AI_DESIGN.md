# Voice AI System Design - "Her"-Like Architecture

> A comprehensive design document for building a fully open-source, real-time voice AI system with voice cloning and Indic language support, running locally on Mac M4.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Ultravox Architecture](#2-ultravox-architecture)
3. [UltraVAD - Endpointing Model](#3-ultravad---endpointing-model)
4. [Voice Cloning Options](#4-voice-cloning-options)
5. [Indic Language Support](#5-indic-language-support)
6. [LLM Routing Strategy](#6-llm-routing-strategy)
7. [Complete "Her"-Like Architecture](#7-complete-her-like-architecture)
8. [Mac M4 Compatibility](#8-mac-m4-compatibility)
9. [Implementation Code](#9-implementation-code)
10. [Training for Indic Languages](#10-training-for-indic-languages)
11. [Future Roadmap](#11-future-roadmap)

---

## 1. Project Overview

### What is Ultravox?

Ultravox is a fast multimodal LLM designed for **real-time voice interactions**. Unlike traditional voice AI pipelines (ASR → LLM → TTS), Ultravox processes audio **directly** into the LLM's embedding space, enabling much faster response times.

### Key Characteristics

- **Multimodal Architecture**: Processes both text AND audio (speech) directly
- **Direct Audio Processing**: Converts audio directly into high-dimensional embedding space
- **Real-time Capable**: Responds much faster than separate ASR + LLM systems
- **Streaming Text Output**: Takes audio input and emits streaming text responses

### Version History

| Version | Date | Notes |
|---------|------|-------|
| v0.7 | Dec 2025 | Latest, GLM-4 default |
| v0.6 | Jun 2025 | Llama 3.3 70B default |
| v0.5 | Feb 2025 | Improved multilingual |
| v0.4.1 | Nov 2024 | Stability improvements |

### Available Models

| Model | Base LLM | Size | Use Case |
|-------|----------|------|----------|
| `fixie-ai/ultravox-v0_7-glm-4_6` | GLM-4 | ~6B | Default v0.7 |
| `fixie-ai/ultravox-v0_6-llama-3_1-8b` | Llama 3.1 | ~8B | Mac M4 recommended |
| `fixie-ai/ultravox-v0_6-llama-3_3-70b` | Llama 3.3 | ~70B | Server deployment |
| `fixie-ai/ultravox-v0_6-gemma3-27b` | Gemma 3 | ~27B | Multilingual focus |
| `fixie-ai/ultravox-v0_6-qwen3-32b` | Qwen 3 | ~32B | Asian languages |

---

## 2. Ultravox Architecture

### Three-Component Design

```
┌─────────────────────────────────────────────────────────────────────┐
│                      ULTRAVOX ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  [Audio Input 16kHz]                                                │
│         │                                                           │
│         ▼                                                           │
│  ┌─────────────────────────────────────────┐                        │
│  │     AUDIO ENCODER (Whisper)             │  ◄── FROZEN            │
│  │     - whisper-large-v3-turbo            │                        │
│  │     - Extracts audio features           │                        │
│  │     - Supports streaming (latency mask) │                        │
│  └─────────────────────────────────────────┘                        │
│         │                                                           │
│         ▼                                                           │
│  ┌─────────────────────────────────────────┐                        │
│  │     MULTIMODAL PROJECTOR                │  ◄── TRAINABLE         │
│  │     - Stacks frames (stack_factor=8)    │      (~5-10M params)   │
│  │     - RMSNorm + Linear + SwiGLU         │                        │
│  │     - Projects to LLM embedding space   │                        │
│  └─────────────────────────────────────────┘                        │
│         │                                                           │
│         ▼                                                           │
│  ┌─────────────────────────────────────────┐                        │
│  │     LANGUAGE MODEL (LLM)                │  ◄── FROZEN            │
│  │     - Llama 3.x / Gemma 3 / GLM-4       │                        │
│  │     - Generates text response           │                        │
│  │     - Streaming output supported        │                        │
│  └─────────────────────────────────────────┘                        │
│         │                                                           │
│         ▼                                                           │
│  [Text Output]                                                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Special Token

- `<|audio|>` marks audio embeddings position in merged sequence
- Audio and text embeddings are merged before LLM processing

### Supported Audio Encoders

| Encoder | Model ID | Notes |
|---------|----------|-------|
| Whisper Large v3 Turbo | `openai/whisper-large-v3-turbo` | Default, best quality |
| Whisper Medium | `openai/whisper-medium` | Faster, smaller |
| Wav2Vec2 | Various | Alternative option |
| Wav2Vec2-BERT | Various | Alternative option |

### Supported LLM Backbones

| LLM | Why Use It |
|-----|------------|
| **Llama 3.1/3.3** | Best reasoning, widely supported |
| **GLM-4** | v0.7 default, good balance |
| **Gemma 3** | Strong multilingual |
| **Qwen 3** | Asian language focus |
| **Mistral** | Fast, efficient |

---

## 3. UltraVAD - Endpointing Model

### What is UltraVAD?

UltraVAD is a **context-aware, audio-native endpointing model** that estimates the probability a speaker has finished their turn. It solves a critical problem: knowing when the user has finished speaking.

### The Problem It Solves

```
Traditional VAD (Voice Activity Detection):
  "I want to order..." (pause) "...a pizza"
  └── Silence detected = WRONG! User still talking

UltraVAD (Context-Aware):
  "I want to order..." (pause) "...a pizza"
  └── Understands context = CORRECT! Waits for user
```

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      UltraVAD                                   │
├─────────────────────────────────────────────────────────────────┤
│  Conversation History    +    User Audio (16kHz)                │
│  (previous turns)             (with prosody cues)               │
│           │                          │                          │
│           ▼                          ▼                          │
│   ┌─────────────┐           ┌────────────────┐                  │
│   │ Text Embed  │           │ Whisper Encoder│                  │
│   └─────────────┘           └────────────────┘                  │
│           │                          │                          │
│           └──────────┬───────────────┘                          │
│                      ▼                                          │
│            ┌─────────────────┐                                  │
│            │ Llama-8B (0.7B) │                                  │
│            │  (post-trained) │                                  │
│            └─────────────────┘                                  │
│                      │                                          │
│                      ▼                                          │
│           P(<|eot_id|>) = 0.85  → End of Turn!                  │
└─────────────────────────────────────────────────────────────────┘
```

### Specifications

| Spec | Value |
|------|-------|
| **Model Size** | 0.7B parameters |
| **Tensor Type** | BF16 |
| **Languages** | 26 (en, fr, de, es, zh, ja, hi, ar, etc.) |
| **Latency** | 65-110ms on A6000 GPU |
| **Output** | Probability [0-1] that user finished |

### Performance vs Competition

| Metric | UltraVAD | Smart-Turn V2 |
|--------|----------|---------------|
| **Accuracy** | **77.5%** | 63.0% |
| **F1-Score** | **81.3%** | 68.1% |
| **AUC** | **89.6%** | 70.0% |

### Usage Pattern

```
┌──────────────┐     Short Silence     ┌───────────┐
│ Silero VAD   │ ───────────────────── │ UltraVAD  │
│ (streaming)  │     detected?         │ (context) │
└──────────────┘                       └───────────┘
       │                                     │
       ▼                                     ▼
  Detect speech                      Is user done?
  boundaries                         P(eot) > 0.1?
```

---

## 4. Voice Cloning Options

### Open Source TTS Models (2025)

| Model | Clone Time | Latency | Languages | License | Best For |
|-------|------------|---------|-----------|---------|----------|
| **XTTS-v2** | 6 sec | <150ms | 17 | Non-commercial | Best quality |
| **Fish-Speech** | 10-30 sec | ~200ms | 8 | CC-BY-NC | Emotion control |
| **Chatterbox** | 5-10 sec | Fast | Multi | Open | Trending, balanced |
| **Bark** | Short clip | Medium | Multi | MIT | Expressive/creative |
| **GPT-SoVITS** | 5 sec | Medium | Multi | Open | Chinese focus |
| **F5-TTS** | 10 sec | Fast | Multi | Open | New, promising |

### XTTS-v2 Details

- **Voice cloning**: Just 6 seconds of audio needed
- **Cross-language**: Clone voice in one language, speak in another
- **Emotion transfer**: Maintains speaker's emotional characteristics
- **Streaming**: <150ms latency with PyTorch on consumer GPU
- **Languages**: 17 supported including Hindi

### Fish-Speech Features

- **Emotion tags**: `(laugh)`, `(whisper)`, `(sob)` for expressive speech
- **Zero-shot cloning**: 10-30 seconds of reference audio
- **Languages**: English, Japanese, Korean, Chinese, French, German, Arabic, Spanish

### Current TTS in Ultravox Codebase

Location: `ultravox/tools/ds_tool/tts.py`

```python
# Supported implementations (for dataset generation only)
- Azure TTS: 25 English neural voices
- ElevenLabs: 26 voice IDs, multilingual v2 model
```

**Note**: Current TTS is for dataset generation, NOT for model output.

---

## 5. Indic Language Support

### Available Datasets

| Dataset | Languages | Total Samples | Source |
|---------|-----------|---------------|--------|
| **IndicVoices** | 23 Indian | ~3M+ | `fixie-ai/IndicVoices_a` |
| **Kathbath** | 13 | ~800K | `fixie-ai/Kathbath` |
| **Shrutilipi** | 17 | ~2M+ | `fixie-ai/Shrutilipi_a` |
| **SeamlessAlign** | 5 Indic | ~3M | `fixie-ai/SeamlessAlign` |
| **FLEURS** | 11 Indic | ~15K | `google/fleurs` |
| **CommonVoice** | 4 Indic | ~10K | `fixie-ai/common_voice_17_0` |

### Language-Specific Sample Counts

| Language | IndicVoices | Kathbath | Shrutilipi | Total |
|----------|-------------|----------|------------|-------|
| **Hindi** | 143K | 92K | 735K | **~970K** |
| **Tamil** | 263K | 96K | 282K | **~641K** |
| **Telugu** | 185K | 71K | 56K | **~312K** |
| **Bengali** | 212K | 47K | - | **~259K** |
| **Marathi** | 144K | 84K | 336K | **~564K** |
| **Malayalam** | 232K | 45K | 224K | **~501K** |
| **Kannada** | 132K | 67K | 192K | **~391K** |
| **Gujarati** | 34K | 67K | 138K | **~239K** |
| **Punjabi** | 129K | 83K | 21K | **~233K** |
| **Odia** | 142K | 48K | 127K | **~317K** |
| **Urdu** | 162K | 49K | - | **~211K** |

### Additional Languages in IndicVoices

Assamese, Bodo, Dogri, Kashmiri, Konkani, Maithili, Manipuri, Nepali, Sanskrit, Santali, Sindhi

---

## 6. LLM Routing Strategy

### The Concept

Route queries to different LLMs based on complexity:
- **Simple queries** → Fast local model
- **Complex queries** → Powerful cloud API (Claude/Gemini)

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    HYBRID LLM ROUTING                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  [Audio Input] → [Ultravox] → [Transcription/Understanding]     │
│                                       │                         │
│                                       ▼                         │
│                         ┌─────────────────────────┐             │
│                         │      SMART ROUTER       │             │
│                         │  (Classify complexity)  │             │
│                         └─────────────────────────┘             │
│                              │              │                   │
│                              ▼              ▼                   │
│                   ┌──────────────┐   ┌──────────────┐           │
│                   │ Simple Query │   │Complex Query │           │
│                   │ Local Gemma  │   │ Claude API   │           │
│                   │ Fast: ~100ms │   │ Better: ~500ms│          │
│                   └──────────────┘   └──────────────┘           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### When to Route Where

| Use Case | Route To | Latency |
|----------|----------|---------|
| Voice commands ("turn on lights") | Local | ~100ms |
| Simple Q&A ("what's the weather") | Local | ~100ms |
| Transcription | Local | ~150ms |
| Complex reasoning | Claude/Gemini | ~500ms |
| Code generation | Claude/Gemini | ~500ms |
| Creative writing | Claude/Gemini | ~500ms |

### Speed Comparison

| Scenario | Latency | Quality |
|----------|---------|---------|
| **Full Local (8B)** | ~150-200ms | Good for basic tasks |
| **Full Local (27B)** | ~400-600ms | Better reasoning |
| **Hybrid → Cloud** | ~500-1000ms | Best reasoning |
| **Hybrid → Local** | ~150-200ms | Fast for simple |

---

## 7. Complete "Her"-Like Architecture

### Current vs Target

```
CURRENT ULTRAVOX:
  [Voice] → [Whisper] → [Projector] → [LLM] → [Text]
                                                 ↓
                                         ❌ NO VOICE OUTPUT

TARGET "HER" SYSTEM:
  [Voice] → [Whisper] → [Projector] → [LLM] → [Text]
                                                 ↓
                                         [Voice Cloning TTS]
                                                 ↓
                                         🔊 Personalized Voice
```

### Complete System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    COMPLETE "HER"-LIKE VOICE AI SYSTEM                  │
│                    (Open Source, Mac M4 Compatible)                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────┐                                                        │
│  │ User speaks │ (Hindi, Tamil, English, etc.)                          │
│  │ in any lang │                                                        │
│  └──────┬──────┘                                                        │
│         │                                                               │
│         ▼                                                               │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ 1. SILERO VAD (~1MB)                                            │    │
│  │    - Detects speech/silence                                     │    │
│  │    - Triggers on voice activity                                 │    │
│  │    - Runs continuously, very fast                               │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│         │                                                               │
│         ▼                                                               │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ 2. ULTRAVAD (0.7B) - Smart Endpointing                          │    │
│  │    - "Is user done speaking?"                                   │    │
│  │    - Context-aware turn detection                               │    │
│  │    - Prevents cutting off mid-sentence                          │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│         │                                                               │
│         ▼                                                               │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ 3. ULTRAVOX (8B) - Audio Understanding                          │    │
│  │    - Whisper encoder (multilingual)                             │    │
│  │    - Understands Hindi, Tamil, Telugu, Bengali, etc.            │    │
│  │    - Direct audio-to-understanding (no separate ASR)            │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│         │                                                               │
│         ▼                                                               │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ 4. LLM ROUTER                                                   │    │
│  │    ┌─────────────────┐    ┌─────────────────┐                   │    │
│  │    │ Simple Query?   │    │ Complex Query?  │                   │    │
│  │    │ → Local Gemma   │    │ → Claude API    │                   │    │
│  │    │   (fast)        │    │   (smart)       │                   │    │
│  │    └─────────────────┘    └─────────────────┘                   │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│         │                                                               │
│         ▼                                                               │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ 5. VOICE CLONING TTS (XTTS-v2 / Fish-Speech)                    │    │
│  │    - Clone ANY voice from 6-30 second sample                    │    │
│  │    - Speak in Hindi, Tamil, English with SAME voice             │    │
│  │    - Express emotions: happy, sad, excited                      │    │
│  │    - Multiple personalities possible                            │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│         │                                                               │
│         ▼                                                               │
│  ┌─────────────┐                                                        │
│  │ 🔊 OUTPUT   │ (Personalized voice in user's language)               │
│  └─────────────┘                                                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Multiple Voice Personalities (Like "Her")

```python
# Different AI personalities with different voices
samantha = VoicePersonality(
    name="Samantha",
    voice_sample="voices/samantha.wav",
    personality="warm, curious, playful"
)

priya = VoicePersonality(
    name="Priya",
    voice_sample="voices/priya.wav",  # Hindi voice
    personality="helpful, professional, calm"
)

arjun = VoicePersonality(
    name="Arjun",
    voice_sample="voices/arjun.wav",  # Male Hindi voice
    personality="friendly, knowledgeable"
)
```

---

## 8. Mac M4 Compatibility

### Device Support in Codebase

Location: `ultravox/utils/device_helpers.py`

```python
def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"  # ✅ Mac Metal supported
    else:
        return "cpu"

def default_dtype() -> torch.dtype:
    # macOS Sonoma 14 enabled bfloat16 on MPS
    return torch.bfloat16 if torch.backends.mps.is_available() else torch.float16
```

### Memory Requirements

| Component | Memory | Notes |
|-----------|--------|-------|
| Ultravox 8B | ~16GB | Core model |
| UltraVAD 0.7B | ~1.5GB | Endpointing |
| XTTS-v2 | ~2GB | Voice cloning |
| Silero VAD | ~50MB | Speech detection |
| **Total** | **~20GB** | Fits in 24GB M4 Pro |

### Mac M4 Variants Compatibility

| Mac M4 Variant | RAM | Compatible Models |
|----------------|-----|-------------------|
| M4 (base) | 16GB | UltraVAD + small models only |
| M4 Pro | 24GB | ✅ Full 8B system |
| M4 Pro | 48GB | ✅ Full system + larger models |
| M4 Max | 64GB+ | ✅ Everything including 27B |

### Requirements

- **macOS**: Sonoma 14+ (for bfloat16 MPS support)
- **Python**: 3.11+
- **Disk**: ~20GB for model weights

### What Works on Mac

| Feature | Status |
|---------|--------|
| Inference with 6-8B models | ✅ Works great |
| UltraVAD endpointing | ✅ Works great |
| Voice cloning (XTTS) | ✅ Works |
| Data processing | ✅ Works |
| Training (small models) | ⚠️ Limited |
| Training (large models) | ❌ Need GPU cluster |

---

## 9. Implementation Code

### Basic Setup

```bash
# 1. Install system dependencies (Mac)
brew install just ffmpeg
brew install pyenv
pyenv install 3.11 && pyenv global 3.11

# 2. Clone and setup
git clone https://github.com/fixie-ai/ultravox
cd ultravox
just install

# 3. Install voice cloning (XTTS)
pip install TTS
```

### Simple Inference

```python
from ultravox.inference import UltravoxInference
from ultravox.data import VoiceSample

# Load model (downloads ~16GB first time)
infer = UltravoxInference(
    model_path='fixie-ai/ultravox-v0_6-llama-3_1-8b',
    device='mps'  # Use Metal on Mac
)

# Run inference
sample = VoiceSample.from_prompt_and_file(
    'Transcribe this audio:',
    'path/to/audio.wav'
)
result = infer.infer(sample)
print(result.text)
```

### UltraVAD Usage

```python
import transformers
import torch
import librosa

# Load UltraVAD
pipe = transformers.pipeline(
    model='fixie-ai/ultraVAD',
    trust_remote_code=True,
    device="mps"
)

# Load audio
audio, sr = librosa.load("user_speech.wav", sr=16000)

# Provide conversation context
turns = [
    {"role": "assistant", "content": "What would you like to order?"},
]

# Get end-of-turn probability
inputs = {"audio": audio, "turns": turns, "sampling_rate": sr}
model_inputs = pipe.preprocess(inputs)

with torch.inference_mode():
    output = pipe.model.forward(**model_inputs, return_dict=True)

# Extract probability
logits = output.logits
audio_pos = int(
    model_inputs["audio_token_start_idx"].item() +
    model_inputs["audio_token_len"].item() - 1
)
token_id = pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
eot_prob = torch.softmax(logits[0, audio_pos, :].float(), dim=-1)[token_id].item()

# Decision
threshold = 0.1
if eot_prob > threshold:
    print("User finished speaking → Generate response")
else:
    print("User still talking → Wait for more audio")
```

### Complete "Her"-Like System

```python
import torch
import anthropic
from ultravox.inference import UltravoxInference
from TTS.api import TTS

# ============ SETUP MODELS ============

# Ultravox for audio understanding
ultravox = UltravoxInference(
    model_path='fixie-ai/ultravox-v0_6-llama-3_1-8b',
    device='mps'
)

# XTTS for voice cloning
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
tts.to("mps")

# Claude for complex queries
claude = anthropic.Anthropic()

# Voice sample for cloning
SPEAKER_WAV = "samantha_voice_sample.wav"

# ============ VOICE CLONING ============

def speak_like_her(text: str, language: str = "hi") -> bytes:
    """Generate speech in cloned voice"""
    audio = tts.tts(
        text=text,
        speaker_wav=SPEAKER_WAV,
        language=language  # hi, ta, en, etc.
    )
    return audio

# ============ QUERY ROUTER ============

def is_complex_query(text: str) -> bool:
    """Determine if query needs cloud LLM"""
    complex_keywords = [
        'explain', 'analyze', 'compare', 'why', 'how does',
        'write code', 'debug', 'summarize', 'translate'
    ]
    return any(kw in text.lower() for kw in complex_keywords)

def route_to_llm(text: str) -> str:
    """Route to appropriate LLM"""
    if is_complex_query(text):
        # Use Claude for complex reasoning
        response = claude.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": text}]
        )
        return response.content[0].text
    else:
        # Use local model for simple queries
        # Could use Gemma 2B or similar
        return local_llm.generate(text)

# ============ MAIN CONVERSATION ============

def her_conversation(audio_input):
    """Full conversation flow like 'Her' movie"""

    # Step 1: Understand user's audio
    result = ultravox.infer(audio_input)
    user_text = result.text
    print(f"User said: {user_text}")

    # Step 2: Detect language
    detected_lang = detect_language(user_text)

    # Step 3: Generate response
    response = route_to_llm(user_text)
    print(f"AI response: {response}")

    # Step 4: Speak in cloned voice
    audio_output = speak_like_her(response, language=detected_lang)

    return audio_output

# ============ MULTIPLE PERSONALITIES ============

class VoicePersonality:
    """Different AI personalities with different voices"""

    def __init__(self, name: str, voice_sample: str, personality: str):
        self.name = name
        self.voice_sample = voice_sample
        self.personality = personality
        self.system_prompt = f"You are {name}. Your personality: {personality}"

    def speak(self, text: str, language: str = "en"):
        return tts.tts(
            text=text,
            speaker_wav=self.voice_sample,
            language=language
        )

    def respond(self, user_input: str) -> str:
        response = claude.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=self.system_prompt,
            messages=[{"role": "user", "content": user_input}]
        )
        return response.content[0].text

# Create personalities
samantha = VoicePersonality("Samantha", "voices/samantha.wav", "warm, curious, playful")
priya = VoicePersonality("Priya", "voices/priya.wav", "helpful, professional")
arjun = VoicePersonality("Arjun", "voices/arjun.wav", "friendly, knowledgeable")
```

---

## 10. Training for Indic Languages

### Training Configuration

Create: `ultravox/training/configs/indic_focused_config.yaml`

```yaml
# Indic Language Focused Training Config

text_model: "google/gemma-3-27b-it"
audio_model: "openai/whisper-large-v3-turbo"

audio_model_lora_config:
  r: 8

# Heavy focus on Indic languages
train_sets:
  # ===== HINDI (Largest) =====
  - name: indicvoices-hi-transcription
    weight: 50
  - name: indicvoices-hi-continuation
    weight: 50
  - name: kathbath-hi-transcription
    weight: 30
  - name: kathbath-hi-continuation
    weight: 30
  - name: shrutilipi-hi-transcription
    weight: 20
  - name: shrutilipi-hi-continuation
    weight: 20
  - name: seamless-hi-transcription
    weight: 10

  # ===== TAMIL =====
  - name: indicvoices-ta-transcription
    weight: 40
  - name: indicvoices-ta-continuation
    weight: 40
  - name: kathbath-ta-transcription
    weight: 20

  # ===== TELUGU =====
  - name: indicvoices-te-transcription
    weight: 30
  - name: indicvoices-te-continuation
    weight: 30
  - name: kathbath-te-transcription
    weight: 15

  # ===== BENGALI =====
  - name: indicvoices-bn-transcription
    weight: 25
  - name: indicvoices-bn-continuation
    weight: 25

  # ===== MARATHI =====
  - name: indicvoices-mr-transcription
    weight: 25
  - name: shrutilipi-mr-transcription
    weight: 15

  # ===== MALAYALAM =====
  - name: indicvoices-ml-transcription
    weight: 20
  - name: shrutilipi-ml-transcription
    weight: 10

  # ===== KANNADA =====
  - name: indicvoices-kn-transcription
    weight: 20
  - name: shrutilipi-kn-transcription
    weight: 10

  # ===== GUJARATI =====
  - name: indicvoices-gu-transcription
    weight: 15
  - name: kathbath-gu-transcription
    weight: 10

  # ===== PUNJABI =====
  - name: indicvoices-pa-transcription
    weight: 15

  # ===== ODIA =====
  - name: indicvoices-or-transcription
    weight: 15

  # ===== URDU =====
  - name: indicvoices-ur-transcription
    weight: 15

  # ===== ENGLISH (for code-switching) =====
  - name: librispeech-clean-transcription
    weight: 10
  - name: librispeech-clean-continuation
    weight: 10

val_sets:
  - name: fleurs-hi_in-transcription
  - name: kathbath-hi-transcription
  - name: indicvoices-hi-transcription

eval_sets:
  - name: fleurs-hi_in-transcription
  - name: fleurs-ta_in-transcription
  - name: fleurs-te_in-transcription
  - name: fleurs-bn_in-transcription
  - name: commonvoice-hi-transcription

# Training parameters
lr: 2e-4
lr_warmup_steps: 1000
batch_size: 2
grad_accum_steps: 6
max_steps: 20000
val_steps: 0.01
save_steps: 0.1
logging_steps: 100
```

### Training Commands

```bash
# Full training (requires 8xH100 GPUs)
poetry run python -m ultravox.training.train \
    --config_path ultravox/training/configs/indic_focused_config.yaml

# DDP training
TRAIN_ARGS="--config_path ultravox/training/configs/indic_focused_config.yaml"
poetry run python -m ultravox.training.helpers.prefetch_weights $TRAIN_ARGS
poetry run torchrun --nproc_per_node=8 -m ultravox.training.train $TRAIN_ARGS

# Debug run on Mac M4 (small batch)
poetry run python -m ultravox.training.train \
    --config_path ultravox/training/configs/indic_focused_config.yaml \
    --batch_size 1 \
    --grad_accum_steps 8 \
    --device mps \
    --max_steps 1000 \
    --report_logs_to tensorboard
```

### Evaluation

```bash
just eval --config_path ultravox/evaluation/configs/eval_config.yaml
```

---

## 11. Future Roadmap

### Phase 1: Basic Setup (Week 1-2)
- [ ] Set up development environment on Mac M4
- [ ] Test Ultravox inference with English
- [ ] Test UltraVAD endpointing
- [ ] Verify Indic language support (Hindi, Tamil)

### Phase 2: Voice Cloning Integration (Week 3-4)
- [ ] Integrate XTTS-v2 for voice cloning
- [ ] Test voice cloning with Indic languages
- [ ] Create sample voice personalities
- [ ] Build end-to-end demo

### Phase 3: LLM Routing (Week 5-6)
- [ ] Implement query complexity classifier
- [ ] Set up Claude API integration
- [ ] Build routing logic
- [ ] Optimize latency

### Phase 4: Indic Language Training (Week 7-10)
- [ ] Prepare training data pipeline
- [ ] Train custom model with Indic focus
- [ ] Evaluate on Indic benchmarks
- [ ] Fine-tune for specific languages

### Phase 5: Production Polish (Week 11-12)
- [ ] Optimize for real-time performance
- [ ] Add emotion detection
- [ ] Build multiple personalities
- [ ] Create demo application

### Future Enhancements

| Feature | Priority | Complexity |
|---------|----------|------------|
| Speech token output (native TTS) | High | High |
| Emotion detection | Medium | Medium |
| Speaker diarization | Medium | Medium |
| Streaming improvements | High | Medium |
| More Indic languages | Medium | Low |
| Mobile deployment | Low | High |

---

## References

### HuggingFace Models
- [Ultravox v0.7 Collection](https://huggingface.co/collections/fixie-ai/ultravox-v07)
- [Ultravox v0.6 Llama 8B](https://huggingface.co/fixie-ai/ultravox-v0_6-llama-3_1-8b)
- [UltraVAD](https://huggingface.co/fixie-ai/ultraVAD)
- [XTTS-v2](https://huggingface.co/coqui/XTTS-v2)

### GitHub Repositories
- [Ultravox](https://github.com/fixie-ai/ultravox)
- [Coqui TTS](https://github.com/coqui-ai/TTS)

### Documentation
- [Ultravox Docs](https://docs.ultravox.ai)
- [Ultravox Demo](https://demo.ultravox.ai)

### Datasets
- [IndicVoices](https://huggingface.co/datasets/fixie-ai/IndicVoices_a)
- [Kathbath](https://huggingface.co/datasets/fixie-ai/Kathbath)
- [Shrutilipi](https://huggingface.co/datasets/fixie-ai/Shrutilipi_a)
- [FLEURS](https://huggingface.co/datasets/google/fleurs)

---

## Open Source Status

| Component | License | Status |
|-----------|---------|--------|
| Ultravox codebase | MIT | ✅ Fully open |
| Whisper encoder | MIT | ✅ Fully open |
| Llama 3.1 8B | Llama 3 License | ✅ Open weights |
| GLM-4 | Apache 2.0 | ✅ Fully open |
| Gemma 3 | Gemma Terms | ✅ Open weights |
| XTTS-v2 | Coqui Public License | ⚠️ Non-commercial |
| Fish-Speech | CC-BY-NC | ⚠️ Non-commercial |
| Bark | MIT | ✅ Fully open |

---

*Document created: January 2025*
*Last updated: January 2025*
