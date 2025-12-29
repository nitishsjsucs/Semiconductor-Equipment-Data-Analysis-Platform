"""
Semiconductor Equipment Data Analysis - Main Entry Point
AI-Driven Manufacturing Data Processing

This project demonstrates how AI can transform tedious manual data processing
into automated, intelligent workflows for semiconductor equipment data.

Usage:
    python main.py                  # Run full analysis pipeline
    python main.py --dashboard      # Launch interactive dashboard
    python main.py --quick          # Quick analysis with fewer features
"""

import argparse
import sys
from pathlib import Path

# Ensure src is in path
sys.path.insert(0, str(Path(__file__).parent))


def run_pipeline(n_features: int = 50, quick: bool = False):
    """Run the complete analysis pipeline."""
    from src.pipeline import SECOMPipeline
    
    n_features = 20 if quick else n_features
    
    pipeline = SECOMPipeline(
        n_features=n_features,
        imbalance_strategy='smote',
        save_outputs=True
    )
    
    results = pipeline.run()
    return results


def run_dashboard():
    """Launch the Streamlit dashboard."""
    import subprocess
    import sys
    
    print("\nLaunching Equipment Data Analyzer Dashboard...")
    print("   Open your browser at: http://localhost:8501")
    print("   Press Ctrl+C to stop the server\n")
    
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])


def main():
    parser = argparse.ArgumentParser(
        description="Semiconductor Equipment Data Analysis - AI-Driven Approach",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    Run full analysis pipeline
  python main.py --dashboard        Launch interactive dashboard
  python main.py --quick            Quick analysis (20 features)
  python main.py --features 30      Custom number of features
        """
    )
    
    parser.add_argument(
        '--dashboard', '-d',
        action='store_true',
        help='Launch the interactive Streamlit dashboard'
    )
    
    parser.add_argument(
        '--quick', '-q',
        action='store_true',
        help='Run quick analysis with fewer features (20)'
    )
    
    parser.add_argument(
        '--features', '-f',
        type=int,
        default=50,
        help='Number of top features to select (default: 50)'
    )
    
    args = parser.parse_args()
    
    if args.dashboard:
        run_dashboard()
    else:
        results = run_pipeline(n_features=args.features, quick=args.quick)
        
        print("\nAnalysis complete! Check the 'outputs' directory for results.")
        print("   Run 'python main.py --dashboard' for interactive exploration.")


if __name__ == "__main__":
    main()
