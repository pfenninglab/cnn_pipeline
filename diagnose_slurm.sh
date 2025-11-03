#!/bin/bash

# SLURM diagnostic script for cnn_pipeline
# This script helps diagnose SLURM access and permission issues

echo "=========================================="
echo "SLURM DIAGNOSTIC SCRIPT"
echo "=========================================="
echo ""

# Check basic SLURM commands
echo "1. Checking SLURM availability:"
if command -v sbatch &> /dev/null; then
    echo "✓ sbatch is available"
    sbatch --version
else
    echo "✗ sbatch not found"
    exit 1
fi
echo ""

# Check user information
echo "2. User information:"
echo "User: $(whoami)"
echo "Groups: $(groups)"
echo ""

# Check available partitions
echo "3. Available partitions:"
if command -v sinfo &> /dev/null; then
    sinfo -o "%P %A %D %T %N"
else
    echo "✗ sinfo not available"
fi
echo ""

# Check partition access
echo "4. Checking partition access:"
if command -v sacctmgr &> /dev/null; then
    echo "User associations:"
    sacctmgr show user $(whoami) withassoc
else
    echo "✗ sacctmgr not available"
fi
echo ""

# Check current job limits
echo "5. Current job limits:"
if command -v scontrol &> /dev/null; then
    echo "User limits:"
    scontrol show user $(whoami)
else
    echo "✗ scontrol not available"
fi
echo ""

# Check current jobs
echo "6. Current jobs:"
if command -v squeue &> /dev/null; then
    echo "Your current jobs:"
    squeue -u $(whoami)
else
    echo "✗ squeue not available"
fi
echo ""

# Test simple job submission
echo "7. Testing simple job submission:"
echo "Creating test job script..."

cat > test_job.sb << 'EOF'
#!/bin/bash
#SBATCH --job-name=test_job
#SBATCH --time=00:01:00
#SBATCH --output=test_output.log
#SBATCH --error=test_error.log

echo "Test job completed successfully"
date
EOF

echo "Test job script created. Attempting to submit..."
if sbatch test_job.sb; then
    echo "✓ Simple job submission successful"
    echo "Cleaning up test job..."
    scancel -u $(whoami) --name=test_job 2>/dev/null
    rm -f test_job.sb test_output.log test_error.log
else
    echo "✗ Simple job submission failed"
fi
echo ""

# Check specific partition access
echo "8. Testing pfen3 partition access:"
echo "Attempting to submit test job to pfen3 partition..."

cat > test_pfen3.sb << 'EOF'
#!/bin/bash
#SBATCH --partition=pfen3
#SBATCH --job-name=test_pfen3
#SBATCH --time=00:01:00
#SBATCH --output=test_pfen3_output.log
#SBATCH --error=test_pfen3_error.log

echo "Test job on pfen3 completed successfully"
date
EOF

if sbatch test_pfen3.sb; then
    echo "✓ pfen3 partition access successful"
    echo "Cleaning up test job..."
    scancel -u $(whoami) --name=test_pfen3 2>/dev/null
    rm -f test_pfen3.sb test_pfen3_output.log test_pfen3_error.log
else
    echo "✗ pfen3 partition access failed"
fi
echo ""

# Check GPU access
echo "9. Testing GPU access:"
echo "Attempting to submit test job with GPU..."

cat > test_gpu.sb << 'EOF'
#!/bin/bash
#SBATCH --partition=pfen3
#SBATCH --gres=gpu:1
#SBATCH --job-name=test_gpu
#SBATCH --time=00:01:00
#SBATCH --output=test_gpu_output.log
#SBATCH --error=test_gpu_error.log

echo "Test GPU job completed successfully"
nvidia-smi
date
EOF

if sbatch test_gpu.sb; then
    echo "✓ GPU access successful"
    echo "Cleaning up test job..."
    scancel -u $(whoami) --name=test_gpu 2>/dev/null
    rm -f test_gpu.sb test_gpu_output.log test_gpu_error.log
else
    echo "✗ GPU access failed"
fi
echo ""

echo "=========================================="
echo "DIAGNOSTIC COMPLETE"
echo "=========================================="
echo "If you see any ✗ marks above, those indicate potential issues."
echo "Common solutions:"
echo "1. Contact your system administrator about partition access"
echo "2. Check if you need to request GPU resources"
echo "3. Verify your account has sufficient job limits"
echo "4. Try submitting to a different partition"
echo "=========================================="