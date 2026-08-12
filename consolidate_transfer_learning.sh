#!/bin/bash
# Consolidate all transfer learning pattern results into one CSV
#
# Usage:
#   bash consolidate_transfer_learning.sh

RESULTS_DIR="results/transfer_learning"
OUTPUT_FILE="results_transfer_learning_all.csv"

if [ ! -d "$RESULTS_DIR" ]; then
    echo "ERROR: Results directory not found: $RESULTS_DIR"
    exit 1
fi

result_count=$(ls -1 "$RESULTS_DIR"/results_transfer_learning_pattern_*.csv 2>/dev/null | wc -l)
if [ $result_count -eq 0 ]; then
    echo "ERROR: No result CSV files found in $RESULTS_DIR"
    echo "Have you run the experiments yet?"
    exit 1
fi

echo "============================================================"
echo "Consolidating Transfer Learning Results"
echo "============================================================"
echo ""
echo "Found $result_count result files in $RESULTS_DIR"
echo "Output: $OUTPUT_FILE"
echo ""

# Consolidate results
{
    # Write header from baseline
    head -1 "$RESULTS_DIR/results_transfer_learning_pattern_baseline.csv" 2>/dev/null

    # Append all rows from baseline
    tail -n +2 "$RESULTS_DIR/results_transfer_learning_pattern_baseline.csv" 2>/dev/null

    # Append all rows from pattern files (skip headers)
    for file in "$RESULTS_DIR"/results_transfer_learning_pattern_pattern*.csv; do
        [ -f "$file" ] && tail -n +2 "$file" 2>/dev/null
    done
} > "$OUTPUT_FILE"

echo ""
final_count=$(wc -l < "$OUTPUT_FILE")
echo "✓ Consolidated: $final_count lines (including header)"
echo "  Expected: ~1093 lines (1092 experiments + 1 header)"
echo ""

# Show some statistics
echo "============================================================"
echo "Result Summary"
echo "============================================================"
echo ""
echo "Total rows: $(( final_count - 1 ))"
echo ""
echo "Results by fit_dataset:"
echo "  Source-only (fit_dataset=\"\"): $(grep ',""$' "$OUTPUT_FILE" | wc -l)"
echo "  Transfer to chestmnist: $(grep ',chestmnist' "$OUTPUT_FILE" | grep -v ',chestmnist,"' | wc -l)"
echo "  Transfer to pneumoniamnist: $(grep ',pneumoniamnist' "$OUTPUT_FILE" | grep -v ',pneumoniamnist,"' | wc -l)"
echo ""

# Check for any missing or failed rows
if grep -q "ERROR\|FAIL\|ERROR" "$OUTPUT_FILE"; then
    echo "⚠ WARNING: Some errors found in results:"
    grep -c "ERROR\|FAIL" "$OUTPUT_FILE" || true
    echo ""
fi

echo "Next step:"
echo "  python analyze_transfer_results.py \\"
echo "    --results $OUTPUT_FILE \\"
echo "    --ranking source_rankings.csv \\"
echo "    --transfer-losses transfer_losses.csv"
echo ""
