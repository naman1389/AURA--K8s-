#!/bin/bash

# Test AI-based remediation with multiple issues
# Tests Ollama -> Gemini fallback chain

set -e

MCP_URL="http://localhost:8000"
API_KEY="test"

echo "=== AI Remediation Testing Suite ==="
echo ""

# Test cases
test_cases=(
    '{"issue_id":"test-001","pod_name":"test-pod-1","namespace":"default","issue_type":"high_memory","severity":"high","description":"Memory usage at 85%"}'
    '{"issue_id":"test-002","pod_name":"test-pod-2","namespace":"default","issue_type":"OOMKilled","severity":"critical","description":"Pod killed due to OOM"}'
    '{"issue_id":"test-003","pod_name":"test-pod-3","namespace":"default","issue_type":"CrashLoopBackOff","severity":"high","description":"Pod in crash loop"}'
    '{"issue_id":"test-004","pod_name":"test-pod-4","namespace":"default","issue_type":"high_cpu","severity":"medium","description":"CPU usage at 90%"}'
    '{"issue_id":"test-005","pod_name":"test-pod-5","namespace":"default","issue_type":"ImagePullBackOff","severity":"medium","description":"Failed to pull image"}'
)

passed=0
failed=0

for i in "${!test_cases[@]}"; do
    test_num=$((i+1))
    test_case="${test_cases[$i]}"
    
    echo "--- Test $test_num ---"
    issue_type=$(echo "$test_case" | python3 -c "import sys, json; print(json.load(sys.stdin).get('issue_type', 'unknown'))")
    echo "Issue Type: $issue_type"
    
    response=$(curl -s -X POST "$MCP_URL/v1/analyze-with-plan" \
        -H "Content-Type: application/json" \
        -H "X-API-Key: $API_KEY" \
        -d "$test_case")
    
    # Check if response contains actions
    if echo "$response" | python3 -c "import sys, json; data=json.load(sys.stdin); exit(0 if 'actions' in data and len(data.get('actions', [])) > 0 else 1)" 2>/dev/null; then
        actions_count=$(echo "$response" | python3 -c "import sys, json; print(len(json.load(sys.stdin).get('actions', [])))")
        reasoning=$(echo "$response" | python3 -c "import sys, json; print(json.load(sys.stdin).get('reasoning', '')[:80])")
        confidence=$(echo "$response" | python3 -c "import sys, json; print(json.load(sys.stdin).get('confidence', 0))")
        source=$(echo "$reasoning" | grep -o "\[.*\]" || echo "[Unknown]")
        
        echo "✅ PASSED"
        echo "   Actions: $actions_count"
        echo "   Confidence: $confidence"
        echo "   Source: $source"
        echo "   Reasoning: $reasoning..."
        ((passed++))
    else
        echo "❌ FAILED"
        echo "   Response: $response" | head -3
        ((failed++))
    fi
    echo ""
done

echo "=== Test Summary ==="
echo "Passed: $passed"
echo "Failed: $failed"
echo "Total: ${#test_cases[@]}"
echo ""

if [ $failed -eq 0 ]; then
    echo "✅ All tests passed!"
    exit 0
else
    echo "❌ Some tests failed"
    exit 1
fi

