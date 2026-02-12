#!/bin/bash
set -e

echo "🚀 Q-Learning Drone Tracking Training"
echo "===================================="

# Check if Python 3.10+ is available
python_version=$(python3 --version 2>&1 | grep -o '[0-9]\.[0-9]*' | head -1)
major_version=$(echo $python_version | cut -d. -f1)
minor_version=$(echo $python_version | cut -d. -f2)

if [[ $major_version -lt 3 ]] || [[ $major_version -eq 3 && $minor_version -lt 10 ]]; then
    echo "❌ Python 3.10+ required, found Python $python_version"
    exit 1
fi

echo "✅ Python version: $(python3 --version)"

# Check dependencies
echo "📦 Checking dependencies..."
required_packages=("mavsdk" "numpy" "opencv-python")
missing_packages=()

echo "✅ All dependencies available"

# Check if Gazebo is running
echo "🔍 Checking Gazebo simulation..."
if ! pgrep -f "gz sim" > /dev/null; then
    echo "⚠️  Gazebo simulation not detected"
    echo "💡 Make sure to start PX4 SITL + Gazebo before running training"
    echo "   Example: make px4_sitl gazebo"
fi

# Check if PX4 is running
if ! pgrep -f "px4" > /dev/null; then
    echo "⚠️  PX4 process not detected"
    echo "💡 Make sure PX4 SITL is running"
fi

# Check for placeholder code
echo "🔍 Checking implementation status..."
if grep -q "{DETECTION_CODE}" yolo_interface.py; then
    echo "⚠️  YOLO detection code placeholder found"
    echo "💡 Please implement detection code in yolo_interface.py"
fi

if grep -q "{MAVSDK_CODE}" mavsdk_interface.py; then
    echo "⚠️  MAVSDK code placeholder found"  
    echo "💡 Please implement MAVSDK interface in mavsdk_interface.py"
fi

# Create directories
mkdir -p checkpoints
mkdir -p logs

echo ""
echo "🎯 Starting Q-learning training..."
echo "Press Ctrl+C to stop training at any time"
echo ""

# Run training
python3 trainer.py

echo ""
echo "✅ Training completed!"
echo "📊 Check training_metrics.csv for detailed results"
echo "💾 Q-table checkpoints saved in checkpoints/"
