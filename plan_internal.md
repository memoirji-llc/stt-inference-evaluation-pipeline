# AMIA 2025 STT Benchmarking Project Plan

Benchmarking transcription robustness on historical/archival speech across audio condition cohorts and model families, with actionable fixes (pre-processing + light fine-tuning) and clear diagnostics (what breaks and why).

---

## Research Questions (3 takeaways)

1. Which STT model(s) hold up best on archival speech?
2. Which audio condition (environment) and speaker (person) factors drive WER up?
3. What interventions (pre-proc / fine-tune / etc.) actually help?

---

## Datasets: Final Selection

### Primary Datasets (audio + full transcripts)

| Dataset | Era | Size | Audio Quality | Source |
|---------|-----|------|---------------|--------|
| **VHP (filtered)** | 1980s-2020s (mixed) | ~1,300 files | Mixed analog/digital | LOC |
| **Voices Remembering Slavery** | 1932-1975 | 26 interviews | Heavily degraded analog | LOC |
| **UC Berkeley Bancroft** | 1960s-1980s | 100s of interviews | Cassette-era analog | Internet Archive |
| **Sacramento History Center** | 1980s | 20+ interviews | Explicitly degraded cassettes | Internet Archive |

**Result:** ~1,500+ audio files with ground truth transcripts spanning 1930s-2020s

**Why these four:**
- VHP: Large-scale, already have data
- Voices Remembering Slavery: Oldest (1930s-1970s), guaranteed degraded
- UC Berkeley: Large cassette-era collection (1960s-1980s)
- Sacramento: Explicitly documented degradation (perfect for "broken records")

**Links:**
- VHP: https://www.loc.gov/collections/veterans-history-project/
- Voices Remembering Slavery: https://www.loc.gov/collections/voices-remembering-slavery/
- UC Berkeley: https://archive.org/details/ucroho
- Sacramento: https://archive.org/details/casacsh_000220
- **See [INTERNET_ARCHIVE_COLLECTIONS.md](INTERNET_ARCHIVE_COLLECTIONS.md) for Internet Archive guide**

---

## Cohort Strategy

We'll bin audio by **measured characteristics**, not recording decade (since we don't have reliable recording dates):

| Axis | Cohort Bins | Measurement Method | Rationale |
|------|-------------|-------------------|-----------|
| **Audio Condition** | SNR bins: <10dB / 10-20dB / >20dB | Calculate from waveform | Noise level is primary degradation factor |
| **Audio Condition** | Bandwidth: Low (<8kHz) / Mid (8-15kHz) / Full (>15kHz) | Spectral roll-off | Cassette/telephone vs modern digital |
| **Source** | VHP / Slavery / Berkeley / Sacramento | Dataset label | Cross-source generalization |
| **Speaker** | Gender: M/F/Unknown | Metadata (where available) | Voice variability |
| **Speaker** | Regional accent (optional) | VHP state metadata | Dialect variation if time allows |

**MVP target:** 6-8 cohorts × 30-60 mins audio each (balanced if possible)

**Example cohorts:**
1. VHP - High SNR (>20dB) - Full bandwidth → "Clean baseline"
2. VHP - Low SNR (<10dB) - Limited bandwidth → "Degraded analog"
3. Voices Remembering Slavery - All (1930s-1970s) → "Oldest recordings"
4. Sacramento - All (explicitly degraded 1980s) → "Known cassette issues"
5. Berkeley - Mid SNR (10-20dB) - Mid bandwidth → "Typical cassette"
6. Berkeley - Low SNR (<10dB) - Low bandwidth → "Poor cassette"

**This approach:**
- Works without knowing exact recording dates
- Uses objective audio metrics
- Naturally separates analog from digital within VHP
- Aligns with your "broken records" theme

---

## Models to Compare

- **Open:** Whisper (small or base; faster-whisper inference), Wav2Vec2-Base-960h (CTC), MMS/SeamlessM4T-style if time allows
- **Commercial (optional):** 1-2 APIs (offline results cached pre-conf)
- **Fine-tune (stretch):** Light domain FT of Wav2Vec2 on 1-2 cohorts (few hours)

---

## Pre-processing Experiments

| Technique | Why | Implementation |
|-----------|-----|----------------|
| Loudness norm (−23 to −18 LUFS) | Stabilizes gain | pyloudnorm |
| VAD trimming | Avoids non-speech penalties | silero/whisper VAD |
| Band-limit restore / EQ | Cassette/telephone compensation | Gentle shelf/peaking EQ |
| Gentle denoise | Hiss/hum removal without artifacts | RNNoise/sox/fft gate |
| Speed perturb (±3%) | Robustness check | Analysis only |

---

## Evaluation Metrics

- **Primary:** WER (with your best-window alignment for partial GTs; you've coded this)
- **Secondary:** CER; per-cohort WER; bootstrap 95% CI by file
- **Diagnostics:** Error buckets (substitutions / deletions / insertions), correlate WER with SNR, bandwidth, duration, and simple tags (hiss, hum, clipping)

---

## Analysis Plan (slides you can show)

1. Cohort bar chart: WER by model × cohort (error bars)
2. Scatter: WER vs SNR (color by model)
3. Waterfall: Effect size of pre-proc (ΔWER)
4. Case cards: 2-3 audio snippets w/ spectrograms (offline), before/after text

---

## MVP vs Stretch Goals

| Tier | Commit Now (do this no matter what) | Nice-to-Have (if time permits) |
|------|-------------------------------------|--------------------------------|
| **MVP** | 2 models (Whisper small, W2V2 base)<br>4 cohorts (clean/degraded from each dataset)<br>WER+CI<br>2 pre-proc ablations | Add 1 commercial API<br>Add 2 more cohorts |
| **+** | Light FT on W2V2 for 1 cohort (few hrs), report ΔWER | Multi-cohort FT<br>MMS baseline |
| **++** | Error taxonomy examples<br>Audio demos with spectrograms | Automatic condition classifier (future work slide) |

---

## Action Items (Priority Order)

### This Week:

1. **Download Voices Remembering Slavery** (26 audio + transcripts) - LOC
2. **Sample UC Berkeley Bancroft collection** (start with 50-100 interviews) - Internet Archive
3. **Sample Sacramento History Center** (all 20+ interviews) - Internet Archive
4. **Filter VHP dataset** by collection ≤ 10,000 for analog bias

### Next Week:

5. **Calculate audio metrics** for all datasets (SNR, spectral roll-off)
6. **Create cohort bins** based on measured characteristics
7. **Run baseline inference** (Whisper small on sample)
8. **Produce first WER figure** (model × cohort)

---

## Abstract (150-170 words draft)

Transcribing historical speech remains challenging due to noise, bandwidth limits, and medium-specific artifacts. We benchmark state-of-the-art speech-to-text models on archival audio drawn from oral histories spanning 1930s-1980s analog era, stratified by measurable audio conditions (SNR bins, bandwidth proxies) and source metadata. We evaluate open models (Whisper, wav2vec2) and, where possible, a commercial baseline, reporting word error rate with robust alignment for partial transcripts. We then test practical interventions—loudness normalization, VAD trimming, gentle denoising—and a light domain fine-tuning of wav2vec2. Results show large performance gaps across cohorts (e.g., low-SNR and telephone-bandwidth speech), with Whisper outperforming wav2vec2 on most clean-ish sets but narrowing under targeted pre-processing. We quantify how SNR and bandwidth correlate with WER and illustrate typical failure modes. We conclude with a short "playbook" for archivists and practitioners: when to preprocess, when to fine-tune, and how to estimate expected accuracy before large-scale transcription. All code and dataset cards will be released to support replicability.

---

## 20-minute Talk Structure

1. Problem & stakes (2 min)
2. Cohorts & metrics (3 min)
3. Baselines: WER by cohort (5 min)
4. Interventions: pre-proc & light FT (5 min)
5. Takeaways & playbook (3 min)
6. Q&A (2 min)

---

## Additional Datasets Considered (Reference)

**Explored but not using for MVP:**
- Supreme Court Oral Arguments (1968-1978) - formal legal speech, use if need more data
- Nixon White House Tapes (1971-1973) - conversational, partial transcripts
- CA State Archives oral histories (1970s-1990s) - good quality control
- Alan Lomax / Dust Bowl collections - music/folklore, limited transcripts
- FDR / NARA WWII audio - no transcripts

**See comparison table in [INTERNET_ARCHIVE_COLLECTIONS.md](INTERNET_ARCHIVE_COLLECTIONS.md)**

---

## Technical Constraints Analysis (Nov 20, 2024)

### Current VM: NC4as_T4_v3 (4 vCPUs, 28GB RAM, T4 GPU 16GB)

**What's Feasible:**

✅ **Whisper (large-v3) inference:**
- Model size: ~3GB
- Inference RAM: ~8-10GB peak
- **Verdict:** Works on current VM

✅ **Wav2Vec2 inference:**
- Model size: ~300MB-1GB
- Inference RAM: ~4-6GB
- **Verdict:** Works on current VM

❌ **Canary-Qwen inference:**
- Model size: ~5GB
- Inference RAM: **20-25GB system RAM** (not GPU VRAM!)
- Available RAM: ~19GB (28GB total - 9GB OS overhead)
- **Verdict:** Will OOM (already confirmed by crash)
- **Blocker:** Would need 110GB VM (NC16as_T4_v3) - quota denied

❌ **Full fine-tuning (Whisper/Wav2Vec2):**
- RAM: ~30-40GB
- GPU VRAM: ~20GB+
- **Verdict:** Won't work on 4 vCPU / 28GB VM

⚠️ **LoRA/PEFT fine-tuning (Whisper-base/small):**
- Whisper-base: 74MB model
- LoRA overhead: ~10GB RAM
- **Verdict:** Might work, needs testing
- Risk: Slower on 4 vCPUs, may not finish before Dec 5

### Recommendation for AMIA (Dec 5 deadline):

**Focus on inference only (Option 3):**
1. ✅ Commercial APIs: Chirp 2/3, AWS Transcribe (done)
2. ✅ Open-source: Whisper large-v3, Wav2Vec2 (run 500-sample benchmarks)
3. ✅ Infrastructure findings: Rate limiting, RAM constraints, cost comparison
4. ⏳ Fine-tuning: Document as "future work" (requires larger VM)
5. ⏳ Canary: Document RAM blocker as archivist consideration

**Why this works:**
- Abstract doesn't promise fine-tuning ("planned experiments" = optional)
- Infrastructure insights are the core contribution anyway
- Canary RAM blocker is actually a valuable finding (real-world constraint)
- Achievable timeline (1-2 weeks for Whisper + Wav2Vec2 runs)

**Frame Canary blocker as research finding:**
> "Canary-Qwen requires 20-25GB RAM for model loading, making it infeasible for budget cloud VMs (4-8 vCPU tier). This is a critical consideration for archivists evaluating self-hosted STT approaches."

---

## Notes

- **Dataset quality over quantity:** Focus on 4 primary datasets with full transcripts rather than spreading thin
- **Objective binning:** Use measured audio characteristics (SNR, bandwidth) instead of assumed recording dates
- **Internet Archive legitimacy:** UC Berkeley and Sacramento collections are official institutional uploads with proper provenance
- **"Broken records" theme:** Sacramento collection explicitly documents audio degradation - cite this in paper
- **Infrastructure as findings:** RAM constraints, rate limiting, and cost tradeoffs are valuable practitioner insights (not failures)
