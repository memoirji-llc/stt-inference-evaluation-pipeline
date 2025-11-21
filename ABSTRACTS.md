# Transcribing a Broken Record: Benchmarking STT for Archival Oral Histories

**Status:** Updated Nov 19, 2024 (reflects actual project scope post-infrastructure blockers)

---

## Abstract (Current - For AMIA 2025)

Modern speech-to-text models promise to unlock archival audio collections, but how well do they perform on degraded historical recordings? This presentation benchmarks commercial and open-source STT systems on oral history collections from the Library of Congress Veterans History Project, focusing on recordings with analog-era audio degradation. We compare commercial APIs with open-source models across the samples using full-duration audio. The session presents practical insights for archivists building transcription pipelines: system requirements and limitations, cost-performance tradeoffs, and decision frameworks for choosing between commercial services versus self-hosted open-source approaches. All code, configurations, and evaluation frameworks will be open-sourced.

**Session Format:** 20-minute presentation + 5-minute Q&A

---

## Goals (Updated)

### Primary Goals (Completed):
1. ✅ **Benchmark commercial APIs** at scale (Chirp 2/3, AWS Transcribe) on 500-sample VHP dataset
2. ✅ **Build production-grade inference pipeline** handling rate limiting, concurrency, memory constraints
3. ✅ **Document real-world implementation challenges** (API quotas, error handling, resource management)
4. ✅ **Cost-benefit analysis** for commercial services (API costs, time, accuracy tradeoffs)

### Secondary Goals (In Progress):
5. ⏳ **Benchmark open-source models** (Whisper, Wav2Vec2, Canary-Qwen) - blocked by GPU memory constraints
6. ⏳ **Test preprocessing interventions** (noise reduction, normalization, filtering)
7. ⏳ **Lightweight fine-tuning** for domain adaptation (resource-dependent)

### Framework Goals:
8. ✅ **Create reproducible evaluation framework** with wandb logging, parquet outputs, modular configs
9. ✅ **Document infrastructure requirements** for archivists (compute, memory, API quotas)

---

## Dataset (Actual)

### Primary Source:
- **Library of Congress Veterans History Project (VHP)**
  - Pre-2010 oral history interviews (analog-era audio degradation)
  - 500-file sample (deterministic `random_state=42`)
  - Full-duration audio (not clipped to 1-5 minutes)
  - Parquet metadata: collection_number, file_id, duration_sec, ground_truth transcripts

### Curation:
- Azure Blob Storage for audio files (`loc_vhp/` prefix)
- Index mapping to preserve filtered dataset blob paths
- Audio preprocessing: 16kHz mono WAV, optional duration trimming

### Ground Truth:
- Human-generated transcripts from LOC metadata
- Used for WER/CER evaluation via `jiwer`
- Normalization options: default (contraction expansion) or Whisper normalizer

---

## Experimental Setup (Actual)

| Step | Description | Status | Notes |
|------|-------------|--------|-------|
| 1 | Data preparation | ✅ Complete | 500-sample VHP parquet with azure_blob_index mapping |
| 2 | Commercial API inference | ✅ Complete | Chirp 2 (497 files), Chirp 3 (500 files), AWS Transcribe (500 files) |
| 3 | Rate limiting handling | ✅ Complete | GCP: 0.6s operation polling + retry; AWS: dynamic throttling |
| 4 | Open-source inference | ⏳ In progress | Blocked by Canary (GPU memory), Whisper/Wav2Vec2 doable |
| 5 | Preprocessing experiments | ❌ Not started | Waiting on compute resources |
| 6 | Fine-tuning | ❌ Not started | Resource + time constraints |
| 7 | Evaluation & metrics | ✅ Partial | WER/CER via jiwer, per-file evaluation_results.parquet |
| 8 | Wandb logging | ✅ Complete | Inference runs logged with experiment_id grouping |

---

## Models Tested

### Commercial APIs (Completed):
| Model | Status | Sample Size | Notes |
|-------|--------|-------------|-------|
| **Google Chirp 2** | ✅ Complete | 500 files | Encountered "Operation requests" quota (150/min) |
| **Google Chirp 3** | ✅ Complete | 500 files | Latest multilingual model |
| **AWS Transcribe** | ✅ Complete | 500 files | Hit 100 concurrent job limit, used dynamic throttling |

### Open-Source Models (Status):
| Model | Status | Sample Size | Notes |
|-------|--------|-------------|-------|
| **Whisper (large-v3)** | ⏳ Pending | 500 planned | Can run on current VM (T4 GPU) |
| **Wav2Vec2** | ⏳ Pending | 500 planned | Can run on current VM |
| **Canary-Qwen-2.5B** | ❌ Blocked | 500 planned | Needs 20-25GB RAM, VM has 28GB (OOM crash), waiting Azure quota |

---

## Key Infrastructure Findings (Research Contribution)

### 1. GCP Chirp Rate Limiting (TWO Quotas):
- **BatchRecognize quota:** 150/min (for starting jobs)
- **Operation requests quota:** 150/min (for checking status) - **commonly missed**
- **Solution:** Batch operations (100/batch) + 60s delays + 0.6s operation polling + exponential backoff retry

### 2. AWS Transcribe Concurrency:
- **Limit:** 100 concurrent jobs
- **Naive approach fails:** Submitting 500 jobs at once → 248 failures
- **Solution:** Dynamic throttling (submit 80 initially, monitor running jobs every 15s, submit more as slots free)

### 3. Memory Optimization:
- **Canary-Qwen-2.5B:** Needs 20-25GB system RAM during model loading (not GPU VRAM)
- **T4 VM (28GB RAM):** Insufficient with OS overhead (~9GB free)
- **Blocker:** Azure free subscription quota limits (rejected upgrade to 110GB VM)

### 4. Cost-Benefit Tradeoffs:
- **Commercial APIs:** Higher cost, no infrastructure management, immediate scale
- **Self-hosted open-source:** Lower marginal cost, requires GPU/RAM, rate limiting still applies (cloud provider quotas)

---

## Deliverables

### Completed:
- ✅ **Inference pipelines** for Chirp 2/3, AWS Transcribe, Whisper, Wav2Vec2, Canary-Qwen
- ✅ **Rate limiting solutions** documented (code + README)
- ✅ **Evaluation framework** (jiwer-based WER/CER with normalization options)
- ✅ **Wandb integration** for experiment tracking
- ✅ **Modular YAML configs** for reproducible runs
- ✅ **Tutorial notebooks** (whisper, wav2vec2, canary - 1-sample quickstart)
- ✅ **Documentation** (README, CONFIG_GUIDE, CURRENT_STATUS, communication templates)

### In Progress:
- ⏳ **Open-source model benchmarks** (Whisper, Wav2Vec2 pending; Canary blocked)
- ⏳ **Preprocessing experiments** (not started)
- ⏳ **AMIA presentation slides** (pending final results)

### Planned:
- 📋 **Archivist decision playbook** (which approach for different scenarios)
- 📋 **Public GitHub release** (code, configs, evaluation framework)
- 📋 **HuggingFace dataset card** (VHP sample metadata, if permitted)

---

## Timeline (Actual)

| Phase | Period | Output | Status |
|--------|---------|---------|--------|
| **Phase 1: Infrastructure** | Sept-Oct 2024 | Azure VM setup, blob storage, data loader | ✅ Complete |
| **Phase 2: Commercial APIs** | Oct-Nov 2024 | Chirp 2/3, AWS Transcribe (500 samples each) | ✅ Complete |
| **Phase 3: Open-source baselines** | Nov 2024 | Whisper, Wav2Vec2, Canary | ⏳ Blocked (Canary), others pending |
| **Phase 4: Experiments** | Nov 2024 | Preprocessing, fine-tuning | ❌ Not started |
| **Phase 5: AMIA presentation** | Dec 5, 2024 | 20-min talk + Q&A | ⏳ In prep (focusing on completed work) |
| **Phase 6: Open-source release** | Dec 2024-Jan 2025 | GitHub repo public, documentation | 📋 Planned |

---

## AMIA Conference Details

- **Conference:** AMIA 2025 Annual Conference
- **Session:** 12205 - "Transcribing a Broken Record: Benchmarking STT for Archival Oral Histories"
- **Date:** December 5, 2024
- **Time:** 3:45-4:15pm (20 min presentation + 5 min Q&A)
- **Location:** Baltimore Marriott Waterfront
- **Presentation approach:** Offline (no wifi dependency), accessibility guidelines followed

---

## Speaker Bio (for AMIA site)

**Arthur Cho** is an AI product builder and independent researcher specializing in conversational AI, NLP, and machine learning evaluation. He holds a Master of Applied Data Science from the University of Michigan and is the Chinese-language publisher of *Designing Machine Learning Systems*. His current work focuses on practical machine learning infrastructure for archival applications, including benchmarking speech-to-text systems for historical audio collections and building tools for oral history preservation.

---

## Research Positioning

### Original Vision:
Comprehensive STT benchmark comparing open-source models (Whisper, Wav2Vec2, MMS) on historical audio with preprocessing and fine-tuning experiments.

### Actual Contribution (Given Constraints):
**Infrastructure-focused research** addressing real-world implementation challenges for archivists:
1. Rate limiting strategies for commercial APIs (GCP's dual quota issue, AWS concurrency)
2. Memory requirements for large speech models (Canary case study)
3. Cost-benefit analysis (commercial vs self-hosted tradeoffs)
4. Decision framework for practitioners (which approach for different scenarios)

### Why This Matters:
Most benchmarking research presents WER comparisons on clean evaluation sets, ignoring the "boring" infrastructure work. Archivists building transcription pipelines need this practical knowledge — API quotas, memory constraints, and cost analysis directly impact their ability to implement STT at scale.

---

## Contingency Plans

### If Canary Unblocks (Azure quota approved):
- ✅ Run 500-sample Canary-Qwen benchmark
- ✅ Add open-source comparison section to presentation
- ✅ Include all three model types in decision framework (commercial, standard open-source, SALM)

### If Canary Stays Blocked:
- ✅ Focus presentation on commercial APIs + infrastructure challenges
- ✅ Run Whisper/Wav2Vec2 as "standard open-source" baseline
- ✅ Acknowledge Canary as future work (include methodology in open-source release)
- ✅ Frame as "Phase 1: Commercial APIs" (implies ongoing research)

### Either Way:
- ✅ Open-source all code/configs (shows methodology is sound)
- ✅ Emphasize infrastructure insights (valuable regardless of model coverage)
- ✅ Publish post-AMIA blog post with full results (extends conference reach)

---

## Notes

- **Presentation format:** Offline, self-contained (no Wi-Fi dependency)
- **Accessibility:** High contrast slides, readable fonts, alt text for visuals
- **Visual aids:** Waveform examples, rate limiting diagrams, cost comparison tables
- **Audience:** Archivists, digital preservationists, AV specialists (prioritize actionable insights over technical deep-dives)
- **Open-source commitment:** All code, configs, and evaluation framework will be public on GitHub post-conference

---

## Lessons Learned (Meta)

1. **Scope creep vs scope reality:** Original vision (comprehensive benchmark) hit infrastructure constraints (Azure quota, memory limits) → pivot to infrastructure-focused contribution
2. **Commercial APIs first:** Running commercial benchmarks first was the right call — they're completed and presentation-worthy on their own
3. **Rate limiting is hard:** GCP's dual quota issue and AWS concurrency required non-trivial engineering (this became a research finding)
4. **Memory ≠ model size:** Canary-Qwen is ~5GB but needs 20-25GB RAM to load (lesson for future model selection)
5. **Honest communication:** Being upfront about blockers in AMIA communications builds credibility (better than overpromising)

---

**Last updated:** November 19, 2024
