"""
Shared file logging utility for all inference and evaluation scripts.

Provides dual logging (console + timestamped file) for better debugging.
"""
import time
from pathlib import Path


class FileLogger:
    """Dual logger that writes to both console and timestamped file."""

    def __init__(self, output_dir, prefix="inference"):
        """Initialize file logger with timestamped filename.

        Args:
            output_dir: Directory to save log file
            prefix: Prefix for log filename (e.g., "inference", "evaluation")
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.log_file = self.output_dir / f"{prefix}_log_{timestamp}.txt"

        # Create log file
        with open(self.log_file, 'w') as f:
            f.write(f"{prefix.capitalize()} Log - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*70 + "\n\n")

        print(f"📝 Logging to: {self.log_file}")

    def log(self, message):
        """Print to console and write to file."""
        print(message)
        with open(self.log_file, 'a') as f:
            # Remove ANSI color codes for file output
            import re
            clean_message = re.sub(r'\x1b\[[0-9;]*m', '', str(message))
            f.write(clean_message + "\n")
            f.flush()  # Ensure immediate write


# Global logger instance (initialized in each script's run() function)
_logger = None


def log(message):
    """Log message to both console and file."""
    if _logger:
        _logger.log(message)
    else:
        print(message)


def init_logger(output_dir, prefix="inference"):
    """Initialize the global logger.

    Args:
        output_dir: Directory to save log file
        prefix: Prefix for log filename

    Returns:
        FileLogger instance
    """
    global _logger
    _logger = FileLogger(output_dir, prefix)
    return _logger
