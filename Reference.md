# Reference

## Key Concepts

### Loss Function:
The most commonly used loss functions with U-Net models for image segmentation include:

- Cross-Entropy Loss: 
 - Used for multi-class segmentation tasks.
 - Measures the dissimilarity between the predicted probability distribution and the ground truth distribution.
- Dice Loss:
 - Particularly effective for imbalanced datasets where one class is significantly more frequent than others.
 - Based on the Dice Coefficient, focusing on the overlap between the predicted segmentation and the ground truth.
- Binary Cross-Entropy Loss:
 - Used for binary segmentation tasks.
 - Measures the dissimilarity between the predicted probability and the binary ground truth.
- Focal Loss:
 - An advanced version of cross-entropy loss that gives more weight to hard-to-classify examples.
  Useful in scenarios where there's a class imbalance.
- Combination of Dice Loss and Cross-Entropy:
 - Often, a combination of Dice loss and cross-entropy (or binary cross-entropy) is used.
 - This combines the advantages of both - general segmentation accuracy from cross-entropy and focus on overlapping regions from Dice loss.
- Tversky Loss:
 - A generalization of Dice loss, allowing more control over false positives and false negatives.
 - Particularly useful in highly imbalanced segmentation tasks.
- Jaccard/Intersection-Over-Union (IoU) Loss:
 - Similar to Dice loss, it measures the overlap between predicted and ground truth segments.
 - Often used in conjunction with other loss functions.
