import sys
import argparse

def main():
    if len(sys.argv) < 2:
        print("IRE Assignment 1 Pipeline Manager")
        print("Usage: python main.py {build|bm25|semantic|evaluate|predict|test} [args...]")
        print("\nCommands:")
        print("  build     - Build the feature store from raw data (and download)")
        print("  build_test- Process and add large test datasets for prediction")
        print("  bm25      - Run BM25 Lexical Retrieval Candidate Generation")
        print("  semantic  - Run Semantic Candidate Generation")
        print("  evaluate  - Run the full Evaluation Harness")
        print("  predict   - Generate Codabench prediction zips")
        print("  test      - Run future-click leakage checks")
        sys.exit(1)
        
    command = sys.argv[1]
    # Remove the command from sys.argv so sub-scripts can parse their own args naturally
    sys.argv.pop(1)
    
    if command == "build":
        from src.pipeline.build import main as build_main
        build_main()
    elif command == "build_test":
        from src.pipeline.process_mind_test import main as mind_test_main
        from src.pipeline.process_ebnerd_test import main as ebnerd_test_main
        mind_test_main()
        ebnerd_test_main()
    elif command == "bm25":
        from src.pipeline.bm25 import main as bm25_main
        bm25_main()
    elif command == "semantic":
        from src.pipeline.semantic import main as semantic_main
        semantic_main()
    elif command == "evaluate":
        from src.pipeline.evaluate import main as eval_main
        eval_main()
    elif command == "predict":
        from src.pipeline.predict import main as predict_main
        predict_main()
    elif command == "test":
        import subprocess
        # Run test_leakage.py as a subprocess so we don't need to refactor it into a function
        subprocess.run([sys.executable, "tests/test_leakage.py"] + sys.argv[1:])
    else:
        print(f"Unknown command: {command}")
        print("Usage: python main.py {build|build_test|bm25|semantic|evaluate|predict|test} [args...]")
        sys.exit(1)

if __name__ == "__main__":
    main()
