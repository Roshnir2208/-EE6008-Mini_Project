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
