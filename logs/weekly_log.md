Week 1 - 13th April to 19th April Contributions
Monday(13th April)
1. Sign language detection project idea - Siddharth
2. Sign boards detection idea - Sharon
3. Recycle waste classification project idea - Sanjay
4. Created a GitHub repository, team members added as collaborators - Roshni
5. Teams channel creation - Sanjay & Sharon

Thursday(16th April)  
1. Found a signboard dataset which could be used for the project - Sharon
2. Found a recycling waste classification dataset - Sanjay

Friday(17th April)
1. Finalised the project ( Recycle waste classification) - Entire team approval
2. Created a document on the Teams channel to add the planned end-to-end workflow - Sanjay
3. Added end-to-end workflow describing the planned implementation - Roshni
4. Added main.py - Sanjay

5. Run 1
Change
Standard CCE loss, narrow backbone (stem=8, max 64 filters), equal class weights, no tricks
Epochs
variable
INT8 size
28.8 KB
val_fg
~0.40
General
~0.30
Recycl.
~0.60
Biodeg.
~0.90
Metal
~0.15
First baseline. Model immediately collapsed to predicting recyclable/biodegradable for everything. Metal recall ≈0.15 — barely better than chance. Class imbalance collapse confirmed.

6. Run 2
Change
First hyperparameter tuning pass on baseline architecture
Epochs
variable
INT8 size
28.8 KB
val_fg
~0.39
General
~0.30
Recycl.
~0.58
Biodeg.
~0.90
Metal
~0.15
Baseline CCE with narrow backbone. Same collapse behaviour as run 1.

7. Run 3
Change
Batch size and epoch count experiments
Epochs
variable
INT8 size
28.8 KB
val_fg
~0.39
General
~0.30
Recycl.
~0.60
Biodeg.
~0.90
Metal
~0.15
Hyperparameter search within baseline CCE setup. No structural improvement.

8. Run 4
Change
Augmentation added, some data rebalancing
Epochs
variable
INT8 size
28.8 KB
val_fg
~0.40
General
~0.30
Recycl.
~0.60
Biodeg.
~0.90
Metal
~0.15
Modest improvement but CCE still allowed dominant classes to dominate the gradient. Core problem persists regardless of augmentation.

9. Run 5
Change
LR schedule experiments on baseline CCE
Epochs
variable
INT8 size
28.8 KB
val_fg
~0.39
General
~0.30
Recycl.
~0.58
Biodeg.
~0.90
Metal
~0.15
LR changes had minimal impact on the fundamental class imbalance collapse. Model still predicted dominant classes.

10. Run 6
Change
Standard CCE, narrow backbone (stem=8, max 64 filters), equal weights
Epochs
variable
INT8 size
28.8 KB
val_fg
~0.40
General
~0.30
Recycl.
~0.60
Biodeg.
~0.90
Metal
~0.15
Model collapsed to predicting recyclable/biodegradable for everything. CCE gave equal weight to all 144 cells so easy classes dominated the gradient. Metal and general classes effectively ignored.

11. Run 7
Change
BG_WEIGHT=0.05 first introduced
Epochs
variable
INT8 size
28.8 KB
val_fg
~0.42
General
~0.40
Recycl.
~0.55
Biodeg.
~0.88
Metal
~0.25
First attempt to address class dominance via background suppression. Strategy was logically sound but factually inapplicable — the dataset had no background-labelled cells.

12. Run 8
Change
BG_WEIGHT=0.1 down-weighting of background cells
Epochs
variable
INT8 size
28.8 KB
val_fg
~0.41
General
~0.39
Recycl.
~0.54
Biodeg.
~0.88
Metal
~0.20
Hypothesis: suppress background gradient so foreground classes get more signal. Failed because whole-image labelling means ALL cells are foreground — no background to suppress.

13. Run 9
Change
BG_WEIGHT experiments continued
Epochs
variable
INT8 size
28.8 KB
val_fg
~0.41
General
~0.40
Recycl.
~0.55
Biodeg.
~0.87
Metal
~0.22
Same structural dead end as runs 7–8. The tweak had no effect because background cells did not exist in the training labels.

14. Run 10
Change
BG_WEIGHT=0.05 suppression, continued refinement
Epochs
variable
INT8 size
28.8 KB
val_fg
~0.42
General
~0.40
Recycl.
~0.55
Biodeg.
~0.88
Metal
~0.25
Marginal improvement over prior runs. Background suppression had no meaningful effect since all 144 cells per image carry the object class in whole-image labelling — no true background cells exist to suppress.
Week 2

1. Collected and explored waste classification datasets for object detection tasks – Sanjay, Sharon
2. Implemented initial object detection pipeline using Edge Impulse (FOMO model) and tested on OpenMV device – Sanjay
3. Successfully deployed and ran inference on OpenMV, but observed low detection performance and missing labels on live feed – Sanjay
4. Identified issues in model performance through confusion matrix analysis (low F1 score ~31%, high background predictions) – Sanjay
5. Discussed limitations of FOMO model for multi-class waste detection (difficulty distinguishing similar classes like plastic, paper, trash) – Team discussion
6. Compared Edge Impulse approach with local TensorFlow Lite implementation, which showed better performance – Sharon
7. Performed dataset cleaning and preprocessing:
8. Removed duplicate images - Team
9. Attempted dataset balancing across classes – Roshni
10. Developed scripts for dataset processing: balance_data.py for class balancing and duplicate removal label_images.py for organizing and labeling dataset using filename  – Roshni
11. Identified key issue in dataset: Images containing multiple different labels (e.g., plastic + glass) causing confusion in training – Team discussion
12. Reprocessed dataset to retain images with single-class labels only to improve model learning – Sanjay
13. Re-uploaded cleaned dataset into Edge Impulse with correct train/test split and bounding box annotations – Sanjay
14. Retrained model with updated dataset and evaluated performance – Sanjay
15. Discussed alternative approaches for improvement: Switching to stronger models (e.g., SSD / YOLO) Considering classification vs detection trade-offs – Team discussion
16. Added extra class ( Organic waste) for Edge Impluse Classification - Roshni
17. Attempted CSV logger via OpenMV IDE using serial logger to pc, but the limitation of lack of SD card, this process could not be done - Roshni

18. Run 11
Change
CLASS_WEIGHTS=[1,3,1,1,4] — general 3×, metal 4×
Epochs
variable
INT8 size
28.8 KB
val_fg
~0.43
General
~0.50
Recycl.
~0.35
Biodeg.
~0.85
Metal
~0.70
First attempt to boost weak classes explicitly. Metal recall improved significantly but recyclable collapsed. Introduced the class weight seesaw problem that persisted through run 13.

19. 
Run 12
Change
CLASS_WEIGHTS=[1,2,1,1,3], manual boosting
Epochs
variable
INT8 size
28.8 KB
val_fg
~0.44
General
~0.48
Recycl.
~0.40
Biodeg.
~0.85
Metal
~0.65
Seesaw worsened. Every improvement to metal recall came at expense of recyclable. The model's single softmax head cannot decouple the two confused pairs simultaneously.

20. Run 13
Change
Adaptive weight callback adjusting every 5 epochs; CLASS_WEIGHTS=[1,3,1,1,4]
Epochs
variable
INT8 size
28.8 KB
val_fg
~0.45
General
0.50
Recycl.
0.35
Biodeg.
0.85
Metal
0.70
Wild oscillation: recyclable 90%→14%→62% across consecutive epochs. Class weight seesaw — boosting metal/general always hurt recyclable/biodeg because the narrow backbone 128-dim GAP vector cannot represent all 4 classes simultaneously at 96×96.

21. Run 14
Change
Loss function replaced: CCE → Focal Loss (γ=2). Equal class weights.
Epochs
variable
INT8 size
28.8 KB
val_fg
~0.55
General
0.61
Recycl.
0.63
Biodeg.
0.70
Metal
0.78
Focal Loss = −(1−p_t)² × log(p_t). The modulating factor (1−p_t)² down-weights easy correct predictions (sea of biodeg cells) and focuses gradient on hard/misclassified samples (general, metal). First run where all 4 classes exceeded 0.60 recall simultaneously.

22. Run 15
Change
Backbone widened: stem 8→16, max filters 64→128
Epochs
variable
INT8 size
63.6 KB
val_fg
~0.56
General
~0.60
Recycl.
~0.60
Biodeg.
~0.75
Metal
~0.73
Wider backbone improved per-class discrimination but pushed size from 28.8→63.6 KB. Still well within 200 KB budget. Focal loss from run 14 retained.

23. Run 16
Change
GlobalAveragePool head replaces spatial conv head; backbone widened
Epochs
variable
INT8 size
63.6 KB
val_fg
~0.57
General
~0.60
Recycl.
~0.60
Biodeg.
~0.78
Metal
~0.72
Replaced spatial Conv2D head with GAP — consistent with whole-image labelling. All 144 output cells derived from same 128-dim vector. Size grew from 28.8→63.6 KB due to backbone widening.

24. Run 17
Change
CosineDecay LR, wide backbone (128 filters), GAP head
Epochs
variable
INT8 size
63.6 KB
val_fg
~0.58
General
~0.62
Recycl.
~0.62
Biodeg.
~0.80
Metal
~0.75
Incremental improvement over run 16. Metal reached 0.75–0.78, general stuck at 0.60–0.65. GAP head forces all 144 cells to share a single prediction vector — good for whole-image labelling but discards location cues.

25. Run 18
Change
CLASS_WEIGHTS=[1,1,1,1,1], FOCAL_GAMMA=2.0, CosineDecay 5e-4→5e-6, custom MinClassAccCheckpoint
Epochs
72 (early stopped at patience=25)
INT8 size
63.6 KB
val_fg
0.608
General
0.645
Recycl.
0.625
Biodeg.
0.848
Metal
0.656
MinClassAccCheckpoint saved the most balanced epoch (min per-class val accuracy improved) rather than best mean. This prevented the model saving an epoch dominated by one easy class. Best balanced run overall across all 19 experiments.

26. Run 19
Change
Fine-tune from Run 18 checkpoint, CLASS_WEIGHTS=[1,1,2,2,2], LR=2e-5
Epochs
14 (early stopped)
INT8 size
63.6 KB
val_fg
0.567 ↓
General
0.581
Recycl.
0.719
Biodeg.
0.818
Metal
0.563
At LR=2e-5 only the 645-param head layer shifts meaningfully. Boosted weights pushed the softmax toward recyclable/metal but broke biodeg separation. Visual confusion between recyclable and metal cannot be resolved at head level — backbone features are identical at 96×96.
