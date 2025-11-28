#!/bin/bash

# Create directories if they don't exist
mkdir -p src models training_results

# Move Python code to src/
for f in id_detection.py token_system.py integrated_system.py inference_simple.py inference_pipeline.py inference_with_augmentation.py augment_dataset.py; do
    if [ -f "$f" ]; then
        mv "$f" src/
        echo "Moved $f to src/"
    else
        echo "Warning: $f not found"
    fi
done

# Move trained weight(s) to models/
if [ -f best.pt ]; then
    mv best.pt models/
    echo "Moved best.pt to models/"
else
    echo "Warning: best.pt not found"
fi

# Move training output images to training_results/
for f in results.png confusion_matrix.png confusion_matrix_normalized.png labels.jpg BoxF1_curve.png BoxPR_curve.png BoxP_curve.png BoxR_curve.png; do
    if [ -f "$f" ]; then
        mv "$f" training_results/
        echo "Moved $f to training_results/"
    else
        echo "Warning: $f not found"
    fi
done

echo "Directory reorganization complete!"
