**Executive Summary**  
Markerless gait analysis has emerged as a cost-effective, accessible, and clinically feasible alternative to traditional marker-based motion capture systems. Leveraging computer vision and deep learning, these systems extract human pose and movement data from video inputs, enabling quantitative assessment of gait parameters without requiring specialized equipment or invasive procedures. Recent studies confirm that markerless methods yield results with high correlation to marker-based systems, particularly in spatiotemporal and joint kinematic parameters, making them suitable for clinical and population-level applications [S11], [S12], [S5].

**Key Findings**  
- Markerless gait analysis using deep learning and computer vision achieves high accuracy in estimating spatiotemporal parameters (e.g., stride length, cadence) and joint angles, with differences from marker-based systems typically below 5° [S5].  
- In clinical settings, markerless systems demonstrate strong correlation with conventional optical tracking systems (OTS), particularly in evaluating gait changes post-treatment (e.g., in idiopathic toe walking) [S3].  
- A 2D markerless methodology using subject-specific lower limb models and reference points (e.g., ankle socks, underwear) enables reliable sagittal plane kinematic analysis with errors within 1–3% of true values [S2].  
- Deep learning-based pose estimation (e.g., DeepLabCut, TCFormer) enables automated detection of anatomical landmarks and joint angle computation, supporting scalable and semi-automated gait analysis [S15], [S16].

**Supporting Evidence**  
- A 2020 study on stroke survivors found no statistically significant differences between markerless and marker-based 2D gait parameters, validating the reliability of markerless methods in clinical domains [S11].  
- A 2022 study comparing 3D markerless and marker-based systems on 16 individuals showed comparable joint angles, with minor underestimation of maximum flexion at the ankle and knee—still within acceptable clinical ranges [S12].  
- A 2024 study on children with idiopathic toe walking demonstrated that markerless analysis accurately captured improvements in ankle sagittal kinematics after corrective casting, confirming its clinical utility [S3].  
- A 2024 study using smartphone-based pose estimation found joint angle differences below 5° compared to marker-based systems, with strong correlations, highlighting the effectiveness of camera placement and device selection [S5].  
- A 2026 study on juvenile idiopathic arthritis used a machine learning pipeline (Mask R-CNN and TCFormer) to extract joint angles from video, validating results against marker-based systems and identifying JIA-related gait abnormalities [S16].  

**Limitations**  
- Performance varies by anatomical region and body morphology; markerless systems show stronger accuracy on well-defined landmarks (e.g., nose, eyes, carpal joints) but struggle with less morphologically discrete areas (e.g., shoulder, hip) [S15].  
- Most studies focus on sagittal plane analysis; 3D gait assessment remains less validated, particularly in diverse populations or across different body types [S12].  
- Accuracy is sensitive to camera placement, orientation, and environmental conditions, requiring standardized protocols for reliable results [S5].  
- While validation against marker-based systems is growing, long-term clinical performance and generalizability across diverse populations and pathologies remain under investigation [S16].

*Note: Evidence is drawn from peer-reviewed studies published between 2011 and 2026. All citations are directly traceable to the provided sources.*

## References
- [S11] Markerless gait analysis in stroke survivors based on computer vision and deep learning (2020). https://openalex.org/W3021102990
- [S2] A 2D Markerless Gait Analysis Methodology: Validation on Healthy Subjects (2015). https://openalex.org/W1996154675
- [S12] Markerless vs. Marker-Based Gait Analysis: A Proof of Concept Study (2022). https://openalex.org/W4220816783
- [S3] Clinical Feasibility of a Markerless Gait Analysis System (2024). https://openalex.org/W4398173279
- [S15] Gait tracking in dogs using DeepLabCut: A markerless machine learning approach for controlled settings (2025). https://openalex.org/W4412749751
- [S5] Improving Gait Analysis Techniques with Markerless Pose Estimation Based on Smartphone Location (2024). https://openalex.org/W4391362407
- [S16] Markerless gait analysis for children and adolescents with juvenile idiopathic arthritis using a machine learning pipeline (2026). https://openalex.org/W7157101352
