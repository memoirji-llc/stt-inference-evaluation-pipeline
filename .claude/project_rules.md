# Project Rules & Conventions

**Internal guidelines for maintaining this repository**

---

## Documentation Organization Rules

### 1. **Root Level = Essential Only**
Only keep files at root if they are:
- User-created working files (`quick_notes.md`, `plan_internal.md`)
- High-level project overviews (`README.md`, `ABSTRACTS.md`)
- Current status tracking (`CURRENT_STATUS.md`)
- Navigation guides (`00_START_HERE.md`)

**Rule:** If it's reference documentation, technical guides, or context/memory docs → put it in `docs/`

---

### 2. **Internal Memory/Context Documents**
Any documents created for tracking project context, decisions, or AI assistance memory should go in:

```
docs/
├── context/           # AI context and memory docs
├── research/          # Historical research
└── [other docs]       # Reference guides
```

**Examples of context docs:**
- Project decision logs
- AI conversation summaries
- Architecture decision records (ADRs)
- Context for future AI sessions

**Rule:** Never create memory/context docs at root level. Always use `docs/context/` or similar.

---

### 3. **Communications Organization**
All AMIA conference-related emails, bios, invitations go in:

```
communications/
├── README.md          # Status of each communication
├── [active files]     # Current drafts/sent emails
└── archive/           # Outdated versions
```

**Rule:** Keep only active communications at top level. Archive superseded versions immediately.

---

### 4. **Naming Conventions**

#### For Dated Documents:
- `REVISED_[name]_[date].md` = Final version, ready to send
- `[name]_YYYY-MM-DD.md` = Dated version if needed

#### For Status Indicators:
- `00_START_HERE.md` = Entry point (sorts first)
- `README.md` in each directory = Explains that directory

#### For Archives:
- `archive/` subdirectory for outdated versions
- Keep original filenames in archive (shows history)

---

### 5. **Configuration Files**
YAML configs for experiments:

```
configs/
├── models/           # Model-specific base configs
└── runs/             # Experiment run configs
```

**Rule:** One config per experiment. Use descriptive names: `vhp-[model]-sample[N]-[duration]-[device].yaml`

---

## File Lifecycle

### Active → Archive → Delete
1. **Active:** File is current and in use
2. **Archive:** File is outdated but kept for reference (move to `archive/` or `docs/research/`)
3. **Delete:** Only if truly redundant and no historical value

**Rule:** When creating a revised version, immediately archive the old one. Don't leave duplicates active.

---

## Directory Structure (As of Nov 20, 2024)

```
amia2025-stt-benchmarking/
├── 00_START_HERE.md              # Navigation guide
├── README.md                      # Main overview
├── ABSTRACTS.md                   # AMIA abstract + research record
├── CURRENT_STATUS.md              # Current project status
├── quick_notes.md                 # User working notes
├── plan_internal.md               # User planning
│
├── communications/                # AMIA emails, bios, invitations
│   ├── README.md
│   ├── [active files]
│   └── archive/
│
├── docs/                          # All reference documentation
│   ├── README.md
│   ├── CONFIG_GUIDE.md
│   ├── QUICK_START.md
│   ├── PROJECT_ROADMAP.md
│   ├── EXPERIMENTS.md
│   ├── [technical docs]
│   ├── context/                   # AI memory/context docs (CREATE THIS IF NEEDED)
│   └── research/                  # Historical research
│
├── configs/                       # YAML configs
│   ├── models/
│   └── runs/
│
├── scripts/                       # Python inference scripts
├── notebooks/                     # Jupyter tutorials
├── outputs/                       # Inference results
├── data/                          # Audio data (not in git)
└── models/                        # Model cache (not in git)
```

---

## When to Create New Docs

### ✅ Create at Root:
- User explicitly creates it
- It's a high-level project summary
- It's a navigation guide

### ✅ Create in docs/:
- Reference guides (how-to, setup)
- Technical documentation
- Internal context/memory docs
- Historical research

### ✅ Create in docs/context/:
- AI conversation summaries
- Decision logs
- Architecture decisions
- Project evolution tracking

### ❌ Never Create at Root:
- Internal memory docs
- Detailed technical guides
- Historical research
- Superseded versions

---

## Cleanup Protocol

### When repo feels messy:
1. **Audit root:** Are there more than 6 markdown files?
2. **Check for duplicates:** Are there multiple docs about the same thing?
3. **Archive old versions:** Move to `archive/` or `docs/research/`
4. **Update READMEs:** Ensure navigation is clear

### Quarterly cleanup:
- Review `archive/` directories
- Delete truly redundant files (with user approval)
- Update navigation guides

---

Last updated: Nov 20, 2024
