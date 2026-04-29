Project Log – Waste Classification System

Week 1 (13th April – 19th April)

Project Ideation & Setup (Monday – Thursday)
1. Proposed initial project ideas:
  a. Sign language detection – Siddharth
  b. Sign board detection – Sharon
  c. Recycle waste classification – Sanjay

2. Final project direction explored collaboratively as a team
3. Created GitHub repository and added collaborators – Roshni
4. Set up Microsoft Teams channel for coordination – Sanjay & Sharon
5. Identified relevant datasets:
  a. Signboard dataset – Sharon
  b. Waste classification dataset – Sanjay

Project Finalisation (Friday)
1. Finalised project: Recycle Waste Classification – Team decision
2. Created and documented planned end-to-end workflow – Sanjay & Roshni
3. Initial project structure setup (main.py and pipeline planning) – Sanjay

Initial Model Experiments (Runs 1–10) – Siddharth (with team support)
1. Implemented baseline model using CCE loss and narrow backbone
2. Conducted multiple experiments:
  a. Hyperparameter tuning
  b. Batch size and epoch adjustments
  c. Learning rate scheduling
  d. Data augmentation trials
3. Identified major issue: class imbalance collapse
  a. Model biased toward recyclable/biodegradable classes
  b. Metal/general classes underperforming
4. Attempted background weighting strategies (Runs 7–10)
  a. Determined ineffective due to dataset structure (no true background cells)

Data Processing & Experiment Support

Roshni:
1. Managed repository setup and collaboration workflows
2. Documented end-to-end pipeline on Teams
3. Cleaned dataset:
  a. Removed duplicate images and classes
  b. Removed/adjusted incorrect annotations and bounding boxes via Python scripts
4. Added missing labels using ImageNet references
5. Developed dataset scripts for balancing and labeling
6. Tested multiple approaches: classification, detection, and transfer learning

Sharon:

1. Performed local model experimentation and debugging
2. Applied image augmentation techniques
3. Conducted hyperparameter tuning and optimization
4. Tested both detection and classification pipelines
5. Worked on model quantization and efficiency improvements
6. Implemented auxiliary features like counters and logging prototypes

Week 2
Dataset Improvement & Model Deployment

Sanjay:

1. Collected and explored datasets for object detection
2. Implemented Edge Impulse (FOMO) pipeline
3. Deployed and tested on OpenMV device
4. Identified performance issues (low F1 ~31%, missing detections)
5. Reprocessed dataset to remove multi-label images
6. Re-uploaded cleaned dataset with correct annotations
7. Retrained and evaluated updated model

Team Contributions:

1. Removed duplicate images collaboratively
2. Attempted dataset balancing – Roshni
3. Identified key dataset issue: multi-class images causing confusion
4. Compared Edge Impulse vs TensorFlow Lite approaches – Sharon
5. Discussed improvements:
  a. Stronger architectures (SSD / YOLO)
  b. Classification vs detection trade-offs

Additional System Work

Roshni:

1. Added new class: Organic Waste
2. Further dataset preprocessing and structuring
3. Attempted CSV logging via serial communication (limited by hardware constraints)

Sharon:

1. Continued model tuning and optimization locally
2. Improved training pipeline stability
3. Experimented with multiple architectures and augmentations
4. Worked on logging and detection/classification refinements

Advanced Model Experiments (Runs 11–19) – Siddharth
1. Introduced class weighting → identified class seesaw problem
2. Implemented adaptive weighting → observed instability
3. Major breakthrough:
  a. Switched CCE → Focal Loss (γ=2) → balanced class performance
4. Improved architecture:
  a. Wider backbone (up to 128 filters)
  b. Global Average Pooling head
5. Applied Cosine Decay learning rate
6. Introduced custom checkpointing (MinClassAccCheckpoint)
7. Achieved most balanced model (Run 18)
8. Fine-tuning (Run 19):
  a. Improved some classes but revealed feature limitations at 96×96 resolution

Week 3 
Ongoing Work Across Team
1. Refining best-performing model from Run 18
2. Improving class separation (especially recyclable vs metal)
3. Evaluating trade-offs between model size and accuracy
4. Exploring further dataset improvements and feature representation
5. Continuing testing on deployment hardware (OpenMV)
6. Stabilising logging and inference pipeline
