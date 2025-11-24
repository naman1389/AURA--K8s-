#!/bin/bash
# Setup Kaggle API credentials as environment variables (backup method)
# This provides an alternative to kaggle.json file

echo "🔧 Setting up Kaggle API environment variables..."
echo "=================================================="

# Your Kaggle credentials (already configured in kaggle.json)
export KAGGLE_USERNAME="namansharma70747"
export KAGGLE_KEY="KGAT_50a1bcd11997cfbff4522407c4fca418"

# Add to shell profile for persistence
SHELL_PROFILE=""
if [ -f "$HOME/.zshrc" ]; then
    SHELL_PROFILE="$HOME/.zshrc"
elif [ -f "$HOME/.bashrc" ]; then
    SHELL_PROFILE="$HOME/.bashrc"
elif [ -f "$HOME/.bash_profile" ]; then
    SHELL_PROFILE="$HOME/.bash_profile"
fi

if [ -n "$SHELL_PROFILE" ]; then
    # Check if already added
    if ! grep -q "KAGGLE_USERNAME" "$SHELL_PROFILE"; then
        echo "" >> "$SHELL_PROFILE"
        echo "# Kaggle API Credentials" >> "$SHELL_PROFILE"
        echo 'export KAGGLE_USERNAME="namansharma70747"' >> "$SHELL_PROFILE"
        echo 'export KAGGLE_KEY="KGAT_50a1bcd11997cfbff4522407c4fca418"' >> "$SHELL_PROFILE"
        echo "✅ Added to $SHELL_PROFILE"
    else
        echo "✅ Already configured in $SHELL_PROFILE"
    fi
else
    echo "⚠️  Shell profile not found. Setting for current session only."
fi

echo ""
echo "✅ Kaggle API environment variables configured!"
echo "   KAGGLE_USERNAME: $KAGGLE_USERNAME"
echo "   KAGGLE_KEY: [hidden]"
echo ""
echo "💡 These will be available in future terminal sessions"
echo "   To use immediately, run: source $SHELL_PROFILE"

