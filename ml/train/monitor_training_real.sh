#!/bin/bash
# Monitor training progress for real datasets (yahoo_s5, numenta)

LOG_FILE="training_real_datasets.log"
CHECK_INTERVAL=30  # Check every 30 seconds

echo "=" | cat
echo "📊 MONITORING TRAINING PROGRESS"
echo "=" | cat
echo ""

while true; do
    # Check if process is running
    PID=$(ps aux | grep "beast_train.py" | grep -v grep | awk '{print $2}')
    
    if [ -z "$PID" ]; then
        echo "❌ Training process not found - may have completed or failed"
        echo "📄 Last 50 lines of log:"
        tail -50 "$LOG_FILE" 2>/dev/null || echo "Log file not found"
        break
    fi
    
    # Get process info
    CPU=$(ps -p "$PID" -o %cpu= 2>/dev/null | tr -d ' ')
    MEM=$(ps -p "$PID" -o %mem= 2>/dev/null | tr -d ' ')
    TIME=$(ps -p "$PID" -o etime= 2>/dev/null | tr -d ' ')
    
    echo "[$(date '+%H:%M:%S')] Process: PID=$PID | CPU=${CPU}% | MEM=${MEM}% | TIME=$TIME"
    
    # Check log for errors
    ERROR_COUNT=$(tail -100 "$LOG_FILE" 2>/dev/null | grep -iE "(error|exception|traceback|failed|fail)" | wc -l | tr -d ' ')
    if [ "$ERROR_COUNT" -gt 0 ]; then
        echo "⚠️  Found $ERROR_COUNT potential errors in last 100 lines"
        echo "Recent errors:"
        tail -100 "$LOG_FILE" 2>/dev/null | grep -iE "(error|exception|traceback|failed)" | tail -5
    fi
    
    # Check for success indicators
    SUCCESS_COUNT=$(tail -50 "$LOG_FILE" 2>/dev/null | grep -iE "(✅|saved|complete|finished|success)" | wc -l | tr -d ' ')
    if [ "$SUCCESS_COUNT" -gt 0 ]; then
        echo "✅ Found $SUCCESS_COUNT success indicators"
    fi
    
    echo ""
    sleep "$CHECK_INTERVAL"
done

