# eval subpackage
"""
ASR Evaluation Module

Core functions for evaluating ASR transcription results.

Usage in notebooks:
    from scripts.eval.analysis import (
        # Text processing
        normalize_text,
        parse_list_column,
        parse_issues_column,

        # WER calculation
        calculate_sample_wer,
        calculate_all_wer,
        get_wer_stats,
        get_error_breakdown,

        # Data quality filtering
        identify_problematic_samples,
        filter_clean_samples,

        # Word-level errors
        extract_error_words,
        aggregate_error_words,

        # Feature analysis
        analyze_categorical_feature,
        analyze_continuous_feature,
        analyze_binary_feature,
        analyze_binary_features,
        analyze_continuous_features,
        expand_issues_to_columns,
        get_correlation_matrix,

        # Constants
        AUDIO_FEATURE_COLS,

        # Data classes
        WERStats,
        ErrorBreakdown,
    )

    from scripts.eval.evaluate import clean_raw_transcript_str
"""

from .evaluate import clean_raw_transcript_str, normalize

from .analysis import (
    # Text processing
    normalize_text,
    parse_list_column,
    parse_issues_column,

    # WER calculation
    calculate_sample_wer,
    calculate_all_wer,
    get_wer_stats,
    get_error_breakdown,

    # Data quality filtering
    identify_problematic_samples,
    filter_clean_samples,

    # Word-level errors
    extract_error_words,
    aggregate_error_words,

    # Feature analysis
    analyze_categorical_feature,
    analyze_continuous_feature,
    analyze_binary_feature,
    analyze_binary_features,
    analyze_continuous_features,
    expand_issues_to_columns,
    get_correlation_matrix,

    # Summary printing
    print_wer_summary,
    print_error_breakdown,

    # Constants
    AUDIO_FEATURE_COLS,

    # Data classes
    WERStats,
    ErrorBreakdown,
    FeatureAnalysis,
)

__all__ = [
    # From evaluate.py
    'clean_raw_transcript_str',
    'normalize',

    # From analysis.py
    'normalize_text',
    'parse_list_column',
    'parse_issues_column',
    'calculate_sample_wer',
    'calculate_all_wer',
    'get_wer_stats',
    'get_error_breakdown',
    'identify_problematic_samples',
    'filter_clean_samples',
    'extract_error_words',
    'aggregate_error_words',
    'analyze_categorical_feature',
    'analyze_continuous_feature',
    'analyze_binary_feature',
    'analyze_binary_features',
    'analyze_continuous_features',
    'expand_issues_to_columns',
    'get_correlation_matrix',
    'print_wer_summary',
    'print_error_breakdown',
    'AUDIO_FEATURE_COLS',
    'WERStats',
    'ErrorBreakdown',
    'FeatureAnalysis',
]
