#!/usr/bin/env python3
"""
ASR Evaluation Analysis Module

Reusable functions for analyzing ASR transcription results.
Designed to be imported into notebooks for reproducible evaluation.

Usage:
    from scripts.eval.analysis import (
        normalize_text,
        calculate_wer_stats,
        calculate_sample_wer,
        identify_problematic_samples,
        filter_clean_samples,
        extract_error_words,
        aggregate_error_words,
        analyze_categorical_feature,
        analyze_continuous_feature,
        analyze_binary_features,
        get_correlation_matrix,
    )
"""

import pandas as pd
import numpy as np
import jiwer
import re
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import Counter
from dataclasses import dataclass, field
from scipy import stats as scipy_stats

# Text normalization - lazy load
_WHISPER_NORMALIZER = None

def _get_whisper_normalizer():
    """Lazy load whisper normalizer."""
    global _WHISPER_NORMALIZER
    if _WHISPER_NORMALIZER is None:
        try:
            from whisper_normalizer.english import EnglishTextNormalizer
            _WHISPER_NORMALIZER = EnglishTextNormalizer()
        except ImportError:
            pass
    return _WHISPER_NORMALIZER


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class WERStats:
    """WER statistics for a dataset."""
    n_samples: int
    mean: float
    median: float
    std: float
    min: float
    max: float
    percentiles: Dict[int, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'n_samples': self.n_samples,
            'mean': self.mean,
            'median': self.median,
            'std': self.std,
            'min': self.min,
            'max': self.max,
            **{f'p{k}': v for k, v in self.percentiles.items()}
        }

    def __repr__(self):
        return (f"WERStats(n={self.n_samples}, mean={self.mean:.3f}, "
                f"median={self.median:.3f}, std={self.std:.3f})")


@dataclass
class ErrorBreakdown:
    """Error breakdown statistics."""
    total_substitutions: int
    total_deletions: int
    total_insertions: int
    total_hits: int
    sub_rate: float  # Mean substitution rate
    del_rate: float  # Mean deletion rate
    ins_rate: float  # Mean insertion rate

    def to_dict(self) -> Dict:
        return {
            'total_substitutions': self.total_substitutions,
            'total_deletions': self.total_deletions,
            'total_insertions': self.total_insertions,
            'total_hits': self.total_hits,
            'sub_rate': self.sub_rate,
            'del_rate': self.del_rate,
            'ins_rate': self.ins_rate,
        }

    def __repr__(self):
        total = self.total_substitutions + self.total_deletions + self.total_insertions
        return (f"ErrorBreakdown(S={self.total_substitutions}, D={self.total_deletions}, "
                f"I={self.total_insertions}, total={total})")


@dataclass
class FeatureAnalysis:
    """Analysis of WER by a categorical or continuous feature."""
    feature_name: str
    feature_type: str  # 'categorical' or 'continuous'
    groups: Optional[Dict[str, Dict]] = None  # For categorical
    correlation: Optional[Dict] = None  # For continuous
    p_value: Optional[float] = None
    significant: bool = False

    def __repr__(self):
        if self.feature_type == 'categorical':
            return f"FeatureAnalysis({self.feature_name}, groups={len(self.groups or {})}, p={self.p_value:.3f})"
        else:
            r = self.correlation.get('spearman_r', 0) if self.correlation else 0
            return f"FeatureAnalysis({self.feature_name}, r={r:.3f}, p={self.p_value:.3e})"


# =============================================================================
# TEXT PROCESSING
# =============================================================================

def normalize_text(text: str, use_whisper: bool = True) -> str:
    """
    Normalize text for WER calculation.

    Args:
        text: Input text
        use_whisper: Use Whisper normalizer (recommended for consistency)

    Returns:
        Normalized text
    """
    if pd.isna(text) or not text:
        return ""

    text = str(text)

    normalizer = _get_whisper_normalizer()
    if use_whisper and normalizer is not None:
        return normalizer(text)

    # Fallback to basic normalization
    tx = jiwer.Compose([
        jiwer.ExpandCommonEnglishContractions(),
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip()
    ])
    return tx(text)


def parse_list_column(val) -> Optional[str]:
    """
    Parse list-like string columns, e.g., "['value']" -> 'value'

    Useful for columns like subject_gender, subject_race from VHP data.
    """
    if pd.isna(val):
        return None
    if isinstance(val, str):
        val = val.strip("[]'\"")
        return val.lower() if val else None
    return None


def parse_issues_column(val) -> List[str]:
    """
    Parse issues string into list.

    Handles format: "['issue1' 'issue2' ...]"
    """
    if pd.isna(val):
        return []
    if isinstance(val, list):
        return val
    val = str(val).strip("[]")
    issues = [re.sub(r'[^a-zA-Z0-9_]', '', issue)
              for issue in val.split("'")
              if issue.strip() and issue.strip() not in [' ', ',']]
    return [i for i in issues if i]


# =============================================================================
# WER CALCULATION
# =============================================================================

def calculate_sample_wer(reference: str, hypothesis: str) -> Optional[Dict]:
    """
    Calculate WER and error breakdown for a single sample.

    Returns:
        Dict with wer, substitutions, deletions, insertions, hits, rates
        or None if calculation fails
    """
    try:
        if pd.isna(reference) or pd.isna(hypothesis):
            return None
        if len(str(reference).strip()) == 0:
            return None

        output = jiwer.process_words(str(reference), str(hypothesis))
        n_ref = len(reference.split())

        if n_ref == 0:
            return None

        wer = (output.substitutions + output.deletions + output.insertions) / n_ref

        return {
            'wer': wer,
            'substitutions': output.substitutions,
            'deletions': output.deletions,
            'insertions': output.insertions,
            'hits': output.hits,
            'sub_rate': output.substitutions / n_ref,
            'del_rate': output.deletions / n_ref,
            'ins_rate': output.insertions / n_ref,
        }
    except Exception as e:
        return None


def calculate_all_wer(df: pd.DataFrame,
                      reference_col: str = 'reference_norm',
                      hypothesis_col: str = 'hypothesis_norm') -> pd.DataFrame:
    """
    Calculate WER for all samples in DataFrame.

    Args:
        df: DataFrame with normalized text columns
        reference_col: Column with normalized reference text
        hypothesis_col: Column with normalized hypothesis text

    Returns:
        DataFrame with WER columns added
    """
    print("Calculating WER for all samples...")

    results = []
    for idx, row in df.iterrows():
        result = calculate_sample_wer(row[reference_col], row[hypothesis_col])
        results.append(result if result else {})

    # Add columns to dataframe
    results_df = pd.DataFrame(results)
    for col in results_df.columns:
        df[col] = results_df[col]

    valid_count = df['wer'].notna().sum()
    print(f"Calculated WER for {valid_count}/{len(df)} samples")

    return df


def get_wer_stats(df: pd.DataFrame, wer_col: str = 'wer') -> WERStats:
    """Calculate summary statistics for WER."""
    wer = df[wer_col].dropna()

    return WERStats(
        n_samples=len(wer),
        mean=wer.mean(),
        median=wer.median(),
        std=wer.std(),
        min=wer.min(),
        max=wer.max(),
        percentiles={
            25: wer.quantile(0.25),
            50: wer.quantile(0.50),
            75: wer.quantile(0.75),
            90: wer.quantile(0.90),
            95: wer.quantile(0.95),
        }
    )


def get_error_breakdown(df: pd.DataFrame) -> ErrorBreakdown:
    """Calculate error breakdown statistics."""
    valid = df[df['wer'].notna()]

    return ErrorBreakdown(
        total_substitutions=int(valid['substitutions'].sum()),
        total_deletions=int(valid['deletions'].sum()),
        total_insertions=int(valid['insertions'].sum()),
        total_hits=int(valid['hits'].sum()),
        sub_rate=valid['sub_rate'].mean(),
        del_rate=valid['del_rate'].mean(),
        ins_rate=valid['ins_rate'].mean(),
    )


# =============================================================================
# DATA QUALITY FILTERING
# =============================================================================

def identify_problematic_samples(
    df: pd.DataFrame,
    min_ref_words: int = 100,
    max_length_ratio: float = 2.0,
    max_wer: float = 1.0,
) -> pd.DataFrame:
    """
    Identify samples with data quality issues.

    These are typically caused by incomplete reference transcripts,
    not ASR failures.

    Args:
        df: DataFrame with WER calculated
        min_ref_words: Flag samples with fewer reference words
        max_length_ratio: Flag samples where hypothesis is much longer
        max_wer: Flag samples with WER above this threshold

    Returns:
        DataFrame with flag columns added
    """
    df = df.copy()

    df['flag_short_ref'] = df['ref_word_count'] < min_ref_words
    df['flag_long_hyp'] = df['length_ratio'] > max_length_ratio
    df['flag_high_wer'] = df['wer'] > max_wer
    df['is_problematic'] = df['flag_short_ref'] | df['flag_long_hyp'] | df['flag_high_wer']

    n_problematic = df['is_problematic'].sum()
    print(f"\nProblematic samples: {n_problematic} ({n_problematic/len(df)*100:.1f}%)")
    print(f"  - Short reference (<{min_ref_words} words): {df['flag_short_ref'].sum()}")
    print(f"  - Long hypothesis (ratio >{max_length_ratio}): {df['flag_long_hyp'].sum()}")
    print(f"  - High WER (>{max_wer}): {df['flag_high_wer'].sum()}")

    return df


def filter_clean_samples(df: pd.DataFrame) -> pd.DataFrame:
    """Return only clean (non-problematic) samples."""
    if 'is_problematic' not in df.columns:
        df = identify_problematic_samples(df)

    df_clean = df[~df['is_problematic']].copy()
    print(f"\nFiltered to {len(df_clean)} clean samples (removed {len(df) - len(df_clean)})")

    return df_clean


# =============================================================================
# WORD-LEVEL ERROR ANALYSIS
# =============================================================================

def extract_error_words(reference: str, hypothesis: str) -> Optional[Dict[str, List[str]]]:
    """
    Extract actual error words from alignment.

    Returns:
        Dict with 'inserted', 'deleted', 'sub_ref', 'sub_hyp' lists
    """
    try:
        output = jiwer.process_words(reference, hypothesis)
        ref_words = reference.split()
        hyp_words = hypothesis.split()

        inserted, deleted, sub_ref, sub_hyp = [], [], [], []

        for chunk in output.alignments[0]:
            if chunk.type == 'insert':
                for i in range(chunk.hyp_start_idx, chunk.hyp_end_idx):
                    if i < len(hyp_words):
                        inserted.append(hyp_words[i])
            elif chunk.type == 'delete':
                for i in range(chunk.ref_start_idx, chunk.ref_end_idx):
                    if i < len(ref_words):
                        deleted.append(ref_words[i])
            elif chunk.type == 'substitute':
                for i in range(chunk.ref_start_idx, chunk.ref_end_idx):
                    if i < len(ref_words):
                        sub_ref.append(ref_words[i])
                for i in range(chunk.hyp_start_idx, chunk.hyp_end_idx):
                    if i < len(hyp_words):
                        sub_hyp.append(hyp_words[i])

        return {
            'inserted': inserted,
            'deleted': deleted,
            'sub_ref': sub_ref,
            'sub_hyp': sub_hyp
        }
    except:
        return None


def aggregate_error_words(df: pd.DataFrame,
                          reference_col: str = 'reference_norm',
                          hypothesis_col: str = 'hypothesis_norm') -> Dict[str, Counter]:
    """
    Aggregate error words across all samples.

    Returns:
        Dict with 'inserted', 'deleted', 'substitutions' Counters
    """
    all_inserted = []
    all_deleted = []
    all_sub_pairs = []

    for idx, row in df.iterrows():
        errors = extract_error_words(row[reference_col], row[hypothesis_col])
        if errors:
            all_inserted.extend(errors['inserted'])
            all_deleted.extend(errors['deleted'])
            all_sub_pairs.extend(zip(errors['sub_ref'], errors['sub_hyp']))

    return {
        'inserted': Counter(all_inserted),
        'deleted': Counter(all_deleted),
        'substitutions': Counter(all_sub_pairs),
    }


# =============================================================================
# FEATURE ANALYSIS
# =============================================================================

def analyze_categorical_feature(
    df: pd.DataFrame,
    feature_col: str,
    wer_col: str = 'wer',
    min_samples: int = 10,
) -> FeatureAnalysis:
    """
    Analyze WER by categorical feature (e.g., gender, race).

    Args:
        df: DataFrame with WER calculated
        feature_col: Categorical feature column
        wer_col: WER column
        min_samples: Minimum samples per group for analysis

    Returns:
        FeatureAnalysis with group statistics and significance test
    """
    df_valid = df[df[wer_col].notna() & df[feature_col].notna()].copy()

    # Get groups with enough samples
    counts = df_valid[feature_col].value_counts()
    valid_groups = counts[counts >= min_samples].index.tolist()

    if len(valid_groups) < 2:
        return FeatureAnalysis(
            feature_name=feature_col,
            feature_type='categorical',
            groups={},
            p_value=None,
            significant=False
        )

    # Calculate stats per group
    groups = {}
    for group in valid_groups:
        wer_vals = df_valid[df_valid[feature_col] == group][wer_col]
        groups[group] = {
            'n': len(wer_vals),
            'mean': wer_vals.mean(),
            'median': wer_vals.median(),
            'std': wer_vals.std(),
        }

    # Statistical test between two largest groups
    sorted_groups = sorted(valid_groups, key=lambda g: groups[g]['n'], reverse=True)
    if len(sorted_groups) >= 2:
        g1, g2 = sorted_groups[0], sorted_groups[1]
        wer1 = df_valid[df_valid[feature_col] == g1][wer_col]
        wer2 = df_valid[df_valid[feature_col] == g2][wer_col]
        try:
            _, p_value = scipy_stats.brunnermunzel(wer1, wer2)
        except:
            p_value = np.nan
    else:
        p_value = np.nan

    return FeatureAnalysis(
        feature_name=feature_col,
        feature_type='categorical',
        groups=groups,
        p_value=p_value,
        significant=p_value < 0.05 if not np.isnan(p_value) else False
    )


def analyze_continuous_feature(
    df: pd.DataFrame,
    feature_col: str,
    wer_col: str = 'wer',
) -> FeatureAnalysis:
    """
    Analyze correlation between continuous feature and WER.

    Args:
        df: DataFrame with WER calculated
        feature_col: Continuous feature column
        wer_col: WER column

    Returns:
        FeatureAnalysis with Spearman correlation
    """
    df_valid = df[df[wer_col].notna() & df[feature_col].notna()].copy()

    if len(df_valid) < 10:
        return FeatureAnalysis(
            feature_name=feature_col,
            feature_type='continuous',
            correlation=None,
            p_value=None,
            significant=False
        )

    r, p = scipy_stats.spearmanr(df_valid[feature_col], df_valid[wer_col])

    return FeatureAnalysis(
        feature_name=feature_col,
        feature_type='continuous',
        correlation={
            'spearman_r': r,
            'n': len(df_valid),
        },
        p_value=p,
        significant=p < 0.05
    )


def analyze_binary_feature(
    df: pd.DataFrame,
    feature_col: str,
    wer_col: str = 'wer',
    min_samples: int = 5,
) -> Dict:
    """
    Analyze WER difference for binary feature (has/doesn't have issue).

    Returns:
        Dict with statistics and significance test
    """
    df_valid = df[df[wer_col].notna()].copy()

    has_feature = df_valid[df_valid[feature_col] == True][wer_col]
    no_feature = df_valid[df_valid[feature_col] == False][wer_col]

    if len(has_feature) < min_samples or len(no_feature) < min_samples:
        return None

    try:
        _, p_value = scipy_stats.brunnermunzel(has_feature, no_feature)
    except:
        p_value = np.nan

    return {
        'feature': feature_col,
        'n_with': len(has_feature),
        'n_without': len(no_feature),
        'wer_with_median': has_feature.median(),
        'wer_without_median': no_feature.median(),
        'wer_with_mean': has_feature.mean(),
        'wer_without_mean': no_feature.mean(),
        'wer_diff': has_feature.median() - no_feature.median(),
        'p_value': p_value,
        'significant': p_value < 0.05 if not np.isnan(p_value) else False
    }


def analyze_binary_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    wer_col: str = 'wer',
    min_samples: int = 5,
) -> pd.DataFrame:
    """
    Analyze WER by multiple binary features.

    Args:
        df: DataFrame with WER calculated
        feature_cols: List of boolean columns to analyze
        wer_col: WER column name
        min_samples: Minimum samples per group

    Returns:
        DataFrame with analysis results sorted by effect size
    """
    results = []
    for col in feature_cols:
        if col not in df.columns:
            continue
        result = analyze_binary_feature(df, col, wer_col, min_samples)
        if result:
            results.append(result)

    if results:
        return pd.DataFrame(results).sort_values('wer_diff', ascending=False)
    return pd.DataFrame()


def expand_issues_to_columns(
    df: pd.DataFrame,
    issues_col: str = 'issues',
    prefix: str = 'has_'
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Expand issues column into boolean columns.

    Args:
        df: DataFrame with issues column
        issues_col: Name of the issues column
        prefix: Prefix for new boolean columns

    Returns:
        Tuple of (DataFrame with new columns, list of new column names)
    """
    df = df.copy()

    # Parse issues into list
    df['_issues_list'] = df[issues_col].apply(parse_issues_column)

    # Get all unique issue types
    all_issues = set()
    for issues in df['_issues_list']:
        all_issues.update(issues)

    # Create boolean columns
    new_cols = []
    for issue in all_issues:
        col_name = f'{prefix}{issue}'
        df[col_name] = df['_issues_list'].apply(lambda x: issue in x)
        new_cols.append(col_name)

    # Clean up temp column
    df = df.drop(columns=['_issues_list'])

    return df, new_cols


def analyze_continuous_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    wer_col: str = 'wer',
) -> pd.DataFrame:
    """
    Analyze correlations between continuous features and WER.

    Args:
        df: DataFrame with features
        feature_cols: List of continuous feature columns
        wer_col: WER column name

    Returns:
        DataFrame with correlation results sorted by absolute correlation
    """
    # Filter to existing columns
    feature_cols = [c for c in feature_cols if c in df.columns]

    results = []
    for col in feature_cols:
        analysis = analyze_continuous_feature(df, col, wer_col)
        if analysis.correlation:
            results.append({
                'feature': col,
                'spearman_r': analysis.correlation['spearman_r'],
                'n': analysis.correlation['n'],
                'p_value': analysis.p_value,
                'significant': analysis.significant,
            })

    if results:
        return pd.DataFrame(results).sort_values('spearman_r', key=abs, ascending=False)
    return pd.DataFrame()


def get_correlation_matrix(
    df: pd.DataFrame,
    cols: List[str],
    method: str = 'spearman'
) -> pd.DataFrame:
    """
    Get correlation matrix for specified columns.

    Args:
        df: DataFrame
        cols: List of columns to include
        method: Correlation method ('spearman' or 'pearson')

    Returns:
        Correlation matrix as DataFrame
    """
    # Filter to existing columns
    cols = [c for c in cols if c in df.columns]
    df_subset = df[cols].dropna()

    return df_subset.corr(method=method)


# Common audio feature columns for convenience
AUDIO_FEATURE_COLS = [
    'snr_db', 'spectral_rolloff_hz', 'spectral_flatness',
    'spectral_centroid_hz', 'zcr_mean', 'zcr_var',
    'loudness_lufs', 'low_freq_energy_ratio', 'duration_sec'
]


# =============================================================================
# SUMMARY PRINTING
# =============================================================================

def print_wer_summary(wer_stats: WERStats, title: str = "WER Summary"):
    """Print formatted WER summary."""
    print(f"\n{title}")
    print("=" * 50)
    print(f"  n:      {wer_stats.n_samples}")
    print(f"  Mean:   {wer_stats.mean:.3f}")
    print(f"  Median: {wer_stats.median:.3f}")
    print(f"  Std:    {wer_stats.std:.3f}")
    print(f"  Min:    {wer_stats.min:.3f}")
    print(f"  Max:    {wer_stats.max:.3f}")
    if wer_stats.percentiles:
        print(f"  Percentiles:")
        for p, val in sorted(wer_stats.percentiles.items()):
            print(f"    {p}th: {val:.3f}")


def print_error_breakdown(err: ErrorBreakdown, title: str = "Error Breakdown"):
    """Print formatted error breakdown."""
    print(f"\n{title}")
    print("=" * 50)
    print(f"  Substitution rate: {err.sub_rate*100:.2f}%")
    print(f"  Deletion rate:     {err.del_rate*100:.2f}%")
    print(f"  Insertion rate:    {err.ins_rate*100:.2f}%")
    print(f"\n  Total counts:")
    print(f"    Substitutions: {err.total_substitutions:,}")
    print(f"    Deletions:     {err.total_deletions:,}")
    print(f"    Insertions:    {err.total_insertions:,}")
    print(f"    Hits:          {err.total_hits:,}")
