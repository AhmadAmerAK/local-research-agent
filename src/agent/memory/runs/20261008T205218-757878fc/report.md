**Executive Summary**  
Markerless human gait analysis has shown promising validity and reliability when compared to marker-based motion capture systems. Multiple studies demonstrate that markerless systems can produce clinically relevant and accurate estimates of spatiotemporal and kinematic gait parameters, particularly in sagittal plane motion. While some discrepancies exist—especially in ankle joint kinematics and specific gait events—overall performance is comparable to the gold standard in many applications. These findings support the potential of markerless systems for use in clinical rehabilitation, research, and field-based settings, though limitations in accuracy and generalizability remain.

---

**Key Findings**  
- Markerless systems achieve good to excellent agreement with marker-based systems for spatiotemporal parameters such as walking speed, step time, and step length [S4].  
- Hip and knee joint angles show moderate to excellent agreement, while ankle joint kinematics exhibit poor concurrent validity and reliability [S4].  
- A 3D markerless system using pose and depth estimations demonstrated high accuracy (LCC > 0.96) and excellent inter-session reliability (RMSE < 3°) for hip and knee angles [S11].  
- A 2D markerless method validated against a clinical gait model showed suitable accuracy (1–3% error) and high correlation (R² between 0.82 and 0.99) for joint kinematics [S2].  
- In stroke survivors, markerless motion capture showed good to excellent agreement for most spatiotemporal parameters, with moderate agreement for non-paretic single-limb support time [S12].  

---

**Supporting Evidence**  
- [S1]: A 2022 study found that a markerless RGB video-based system produced similar spatio-temporal parameters and joint angles to marker-based systems, with slight underestimation of maximum flexion for ankle and knee angles.  
- [S11]: A 2024 study evaluated a 3D markerless system and reported high accuracy (LCC > 0.96) and excellent reliability (RMSE < 3°) for hip and knee joint angles, though moderate-to-high accuracy with biases were observed during specific gait events.  
- [S2]: A 2015 study validated a 2D markerless technique against a clinical gait model, showing high correlation (R² between 0.82 and 0.99) and acceptable error margins (3.9° to 6.1°) for hip, knee, and ankle joint kinematics.  
- [S12]: A 2025 study found good to excellent agreement between markerless motion capture and an instrumented walkway system in stroke survivors, with notable exceptions in single-limb support time.  
- [S4]: A 2024 meta-analysis of 22 studies concluded that markerless systems show good to excellent validity and reliability for spatiotemporal parameters, with moderate-to-excellent agreement for hip and knee joints and poor validity for ankle joints.  

---

**Limitations**  
- Accuracy varies significantly by joint and movement phase, with the ankle joint showing poor concurrent validity and reliability [S4, S3].  
- Most studies involve healthy participants, limiting generalizability to clinical populations such as stroke survivors or individuals with neuromotor disorders [S11, S12].  
- Performance is sensitive to camera configuration, subject positioning, and environmental conditions, particularly in markerless systems relying on OpenPose or depth estimation [S3].  
- Limited key points in pose estimation models and sample rate differences between systems may introduce systematic biases [S11].  
- The evidence base is still evolving, with fragmented validation data across diverse populations and movement tasks [S16].  

*Note: While several studies validate markerless systems against marker-based systems, the overall evidence is strongest for spatiotemporal parameters and sagittal-plane kinematics. Validation for complex or pathological gait patterns remains less robust and requires further investigation.*

## References
- [S1] Markerless vs. Marker-Based Gait Analysis: A Proof of Concept Study (2022). https://openalex.org/W4220816783
- [S11] Validation of a 3D Markerless Motion Capture Tool Using Multiple Pose and Depth Estimations for Quantitative Gait Analysis (2024). https://openalex.org/W4404075327
- [S2] A 2D Markerless Gait Analysis Methodology: Validation on Healthy Subjects (2015). https://openalex.org/W1996154675
- [S12] Validity of AI-Driven Markerless Motion Capture for Spatiotemporal Gait Analysis in Stroke Survivors (2025). https://openalex.org/W4413764071
- [S3] Validation of a 3D Markerless System for Gait Analysis Based on OpenPose and Two RGB Webcams (2021). https://openalex.org/W3162115901
- [S4] Accuracy, Validity, and Reliability of Markerless Camera-Based 3D Motion Capture Systems versus Marker-Based 3D Motion Capture Systems in Gait Analysis: A Systematic Review and Meta-Analysis (2024). https://openalex.org/W4399389708
- [S16] Validations and applications of markerless motion capture using OpenCap: a scoping review (2026). https://openalex.org/W7171835655
