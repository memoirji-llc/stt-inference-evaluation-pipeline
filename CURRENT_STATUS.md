# Current Project Status & AMIA 2025 Research Direction

**Last Updated**: November 19, 2024

---

## What We're Trying to Achieve

**Research Question for AMIA 2025 Workshop Session:**

How do open-source speech-to-text models (Whisper, Canary-Qwen, Wav2Vec2) perform on degraded historical audio from oral history interviews, compared to commercial STT services (Google Chirp, AWS Transcribe), under different conditions?

**Evaluation Dimensions:**
1. **Audio preprocessing**: Raw audio vs. preprocessed (noise reduction, normalization)
2. **Model adaptation**: Baseline models vs. fine-tuned models vs. commercial service adaptation
3. **Dataset characteristics**: Veterans History Project (VHP) pre-2010 interviews - unprofessional recording environments, analog-era audio degradation, conversational speech with elderly speakers

**Target Deliverable**: Reproducible benchmark pipeline demonstrating WER (Word Error Rate) performance across model families on challenging real-world historical audio, with insights into cost-accuracy tradeoffs for cultural heritage digitization projects.

---

## What We've Completed So Far

### ✅ Infrastructure & Pipeline Setup
- **Production-grade inference pipeline** with Azure blob integration, wandb tracking, and file logging
- **Evaluation framework** with whisper normalizer and per-file WER metrics
- **Rate limiting protection** for commercial APIs (GCP 150/min quota, AWS 100 concurrent jobs)
- **Memory optimization** for large-scale batch processing on cloud VMs
- **Data management**: 3,800+ VHP pre-2010 audio files indexed and ready in Azure blob storage

### ✅ Commercial STT Models - Benchmarked (500 samples each)
1. **AWS Transcribe**: 497/500 successful transcriptions
   - Full-duration audio, real production API
   - Implemented dynamic throttling for 100 concurrent job limit

2. **Google Chirp 3**: 497/500 successful transcriptions
   - BatchRecognize API with 150 requests/min rate limiting
   - Operation status polling with 0.6s delays to avoid quota exceeded

3. **Google Chirp 2**: 255/497 successful (initial run hit quotas, fixed for rerun)
   - Same infrastructure as Chirp 3
   - Retry logic with exponential backoff for 429 errors

**Key Achievement**: Successfully handled API rate limits, memory constraints, and large-scale batch processing on cloud infrastructure.

### ✅ Open-Source Models - Partially Benchmarked
1. **Whisper**: Small-scale tests completed (sample runs)
   - Infrastructure ready for 500-sample benchmark
   - Faster-whisper integration for efficiency

2. **Wav2Vec2**: Small-scale tests (10 samples, 300s duration)
   - Baseline established, ready to scale

### ✅ Tutorial Notebooks Created
- `tutorial_infer_whisper.ipynb`: Working notebook for Whisper inference
- `tutorial_infer_wav2vec2.ipynb`: Working notebook for Wav2Vec2 inference
- `tutorial_infer_canary.ipynb`: Prepared but blocked (see below)

---

## Current Impediment: Canary-Qwen Inference Blocked

### The Problem
**Cannot run NVIDIA Canary-Qwen inference** due to Azure VM resource limitations.

**Root Cause**:
- Canary-Qwen-2.5B model requires ~20-25 GB system RAM during loading (peaks at 20GB)
- Current VM: `NC4as_T4_v3` with **28 GB total RAM**
- With OS and other processes using ~9 GB, only ~19 GB available
- Model loading triggers OOM (Out of Memory) crashes, causing entire VM to freeze

**What We've Tried**:
1. ✅ Attempted `canary-1b-flash` (smaller model, only ~12 GB RAM needed)
   - **Failed**: Different model architecture (EncDecMultiTaskModel vs SALM)
   - Our script uses `SALM.from_pretrained()` which doesn't work with canary-1b-flash
   - Would require separate inference script

2. ✅ Implemented GPU memory cleanup (CUDA cache clearing)
   - **Insufficient**: Problem is system RAM, not GPU VRAM

3. ✅ Set HuggingFace cache to local directory (avoid re-downloads)
   - **Helpful but doesn't solve RAM issue**

4. ✅ Disabled wandb for notebook runs (reduce overhead)
   - **Marginal improvement, still OOM**

### Why This Matters for AMIA 2025
**Canary-Qwen is critical** because:
- Recent model (2024) with strong performance on conversational speech
- Represents latest open-source SALM (Speech-Augmented Language Model) architecture
- Needed for fair comparison: Commercial APIs vs. Latest Open-Source vs. Established Open-Source (Whisper)
- According to Artificial Analysis benchmark, Canary performs competitively with commercial services

### Attempted Solution: Azure VM Resize
**Goal**: Upgrade to `NC16as_T4_v3` with 110 GB RAM (sufficient for Canary + future fine-tuning)

**Status**: **BLOCKED - Quota Request Rejected** (Nov 19, 2024)

**Azure Response**:
> "Due to high demand for these graphics-enabled VM types, availability is limited for customers using free/benefit/sponsorship subscriptions. Please consider alternative VM Types that may be suitable for your requirements, or consider creating a new pay-as-you-go subscription."

**Next Steps**:
1. ⏳ Awaiting response to follow-up appeal (emphasized research/academic nature)
2. 🔄 If rejected again: Convert subscription to pay-as-you-go (~$400-600/month for on-demand usage)
3. 📊 Meanwhile: Complete Whisper and Wav2Vec2 full 500-sample benchmarks on current VM

---

## What We're Waiting For

### Immediate Blocker
- **Azure quota approval** for NC16as_T4_v3 (110 GB RAM) to unblock Canary-Qwen inference

### Once Unblocked - Remaining Tasks for AMIA 2025

#### 1. Complete Open-Source Model Benchmarks (500 samples each)
- [ ] Whisper large-v3 (full 500-sample run)
- [ ] Canary-Qwen-2.5B (pending VM upgrade)
- [ ] Wav2Vec2 large (scale from 10 to 500 samples)

#### 2. Audio Preprocessing Pipeline
- [ ] Implement noise reduction pipeline (spectral gating, bandpass filtering)
- [ ] Run all models on preprocessed vs. raw audio
- [ ] Quantify preprocessing impact on WER

#### 3. Model Fine-Tuning (if time permits)
- [ ] Fine-tune Whisper on VHP subset with LoRA
- [ ] Fine-tune Canary-Qwen (requires 110 GB RAM VM)
- [ ] Compare fine-tuned vs. baseline performance

#### 4. Analysis & Paper
- [ ] Statistical analysis of WER distributions across models
- [ ] Cost-performance tradeoffs ($/hour × accuracy)
- [ ] Error analysis: what types of speech do different models struggle with?
- [ ] Recommendations for cultural heritage digitization projects

---

## Technical Debt & Lessons Learned

### Successfully Resolved
1. ✅ **GCP rate limiting**: 150 BatchRecognize requests/min + 150 operation requests/min
   - Solution: Batched submission with 60s delays + 0.6s polling delays

2. ✅ **AWS rate limiting**: 100 concurrent jobs hard limit
   - Solution: Dynamic throttling with running job monitoring

3. ✅ **File logging throughout pipeline**: Timestamped logs for debugging without wandb dependency

4. ✅ **HuggingFace cache management**: Using local `models/` directory instead of `~/.cache`

### Still Outstanding
1. ⚠️ **Canary-Qwen RAM requirements**: Need VM upgrade or model offloading implementation
2. ⚠️ **Audio preprocessing pipeline**: Not yet implemented
3. ⚠️ **Fine-tuning infrastructure**: Will need same 110 GB RAM for training

---

## Project Health Assessment

**Overall Progress**: ~60% complete for AMIA 2025 submission

**On Track**:
- Commercial API benchmarks ✅
- Infrastructure & pipeline ✅
- Small-scale open-source tests ✅

**At Risk**:
- Canary-Qwen benchmark (blocked by Azure quota)
- Audio preprocessing comparison (not started)
- Fine-tuning experiments (time + resources)

**Mitigation Strategy**:
- Proceed with Whisper + Wav2Vec2 benchmarks on current VM
- If quota denied: Pay-as-you-go conversion (~$200-400 additional cost for project duration)
- Prioritize: Full benchmarks > Preprocessing > Fine-tuning (if time allows)

**Timeline Concern**: Conference deadline approaching - need Canary results within 1-2 weeks to allow time for analysis and paper writing.

---

## Contacts & Resources

**Azure Support**: SR #2511190040003655 (quota increase request)
**VM Current**: `NC4as_T4_v3` (28 GB RAM, 1x T4 GPU)
**VM Needed**: `NC16as_T4_v3` (110 GB RAM, 1x T4 GPU)
**Resource Group**: `rg-amia-ml`
**Subscription**: Free/benefit tier (considering pay-as-you-go conversion)
