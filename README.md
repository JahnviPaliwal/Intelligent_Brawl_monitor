# Intelligent_Brawl_Monitor:
This project develops a Real-Time Facial Expression and Pose Detection System using OpenCV. The AI model identifies angry facial expressions and aggressive body poses from live camera feeds, providing timely alerts about potential conflicts to enhance safety in various environments. In simple words, this senses the aggression from any human being and gives an alert as an SOS to the email ids provided. 


- Note: while using this code, make sure you keep the sender's email as your choice mail ID and check. This code can be run in the Google Collab.

## Objective:


- **Real-Time Detection:**
  To develop a system that continuously monitors video feeds for   facial expressions and body poses.
- **Emotion Recognition:** To accurately identify angry expressions using machine learning algorithms trained on a comprehensive dataset.
- **Pose Analysis:** To detect specific aggressive body poses that may indicate impending conflict.
- **Alert Mechanism:** To implement an email alert system that notifies designated recipients when both an angry expression and an aggressive pose are detected, enabling prompt intervention.

## Features:

- **Real-Time Video Processing:**
  - Faster processing (can be expanded to 4K quality which may support electrical parameters as a specific chip or may provide clarity in the processing with examples stored for Deep Learning).
  - Utilizes OpenCV to capture video from a camera in real-time.
  - Processes each frame to detect faces and analyze expressions.
- **Facial Expression Recognition:**
  - Employs machine learning models trained on a diverse dataset of facial expressions.
  - Specifically targets the detection of anger, using features like eyebrow positioning, mouth shape, and eye movement.
  - Uses each point as a node where it symbolizes the pose estimation. 
- **Pose Detection Algorithms:**
  - Integrates multiple models for pose detection (e.g., OpenPose or MediaPipe) to recognize various body postures associated with aggression.
  - Capable of distinguishing between neutral and aggressive stances based on key points of the human body.
- **Automated Email Alerts:**
  - Configured to send real-time email notifications when both conditions (angry expression and aggressive pose) are met.
  - Customizable alert settings allow users to specify recipient email addresses and message content.
- **Dataset Matching:**
  - Compares real-time detections against a pre-trained dataset to ensure high accuracy in emotion recognition.
  - Continuously updates the model with new data to improve detection capabilities over time.
## Technologies Used:
- **Python:** The primary programming language used for data manipulation and model building.
- **Pandas:** A library for data analysis and manipulation, essential for handling the dataset.
- **NumPy:** Utilized for numerical computations and array operations.
- **TensorFlow/Keras:** Frameworks for building and training the regression model.
- **Matplotlib:** Used to visualize the results through graphs.
- **OpenCV:** A powerful open-source computer vision library used for real-time image processing and video analysis. It facilitates facial detection, expression recognition, and pose estimation.
- **Pre-Trained Models:** Utilization of existing models ( OpenPose, MediaPipe) for pose detection to expedite development and improve accuracy.
- **Email Automation Tools:** Libraries like smtplib in Python are used to automate the sending of email alerts when aggressive behavior is detected.
- **Dataset Resources:** Publicly available datasets for facial expressions (FER2013 @Kaggle)
- **Hardware:** A real-time camera setup (webcam or IP camera) to capture video feeds for analysis.



## Applications:

The applications of this system are vast and varied:
- **Security Surveillance:** Deployed in public areas such as malls, schools, or event venues to monitor for potential fights or disturbances.
- **Workplace Safety:** Used in corporate environments to identify aggressive behavior among employees, facilitating early intervention by HR or security personnel.
- **Research and Behavioral Studies:** Valuable for psychologists and sociologists studying human emotions and interactions in controlled experiments or naturalistic settings.
- **Smart Home Systems:** Integrated into home security systems to enhance safety by monitoring household interactions.
## Advantages
- **Proactive Conflict Prevention:** By providing timely alerts, the system enables authorities or responsible individuals to intervene before situations escalate into physical confrontations.
- **High Accuracy in Detection:** With continuous training on diverse datasets, the model can achieve high accuracy in recognizing emotions and poses.
- **Scalability and Flexibility:** The architecture allows for easy integration of additional features, such as recognizing other emotions or poses as more data becomes available.
## Drawbacks
- **Risk of False Positives:** The system may misinterpret benign expressions or pose as aggressive, leading to unnecessary alerts that could cause alarm among users.
- **Dependence on Environmental Conditions:** Performance may be affected by factors such as lighting conditions, camera quality, or occlusions (e.g., objects blocking the view).
- **Privacy Concerns:** Continuous monitoring raises ethical issues regarding privacy rights; individuals may feel uncomfortable being watched without consent
## Example of estimated image output:
![Screenshot 2024-11-04 224305](https://github.com/user-attachments/assets/b4d8d72f-41cd-4fc4-a560-6f9adfc0ccfa)



## Future Modifications and Scopes:
To enhance the functionality and effectiveness of the system, several modifications and developments are planned:
- Expansion of Dataset:
Collecting a broader range of facial expressions and body poses from diverse demographics to improve model robustness.
Incorporating real-world scenarios to train the model on various contexts where aggression might manifest.
- Advanced Machine Learning Techniques:
Exploring deep learning frameworks like TensorFlow or PyTorch for more sophisticated models that can handle complex datasets more effectively.
Implementing techniques such as transfer learning to leverage existing models trained on large datasets for improved performance with limited data.
- Integration with IoT Devices:
Linking the detection system with smart home devices (like alarms or cameras) for comprehensive security solutions that respond automatically to detected aggression.
Developing mobile applications that allow users to receive alerts directly on their smartphones for immediate action.
- In simple terms from an Indian perspective, motivation to alert in rural areas and women's safety. It can also be kept handy in terms of pocket-tronics for SOS and enhancing better chances of empowerment in technology and education.
- Further can be used for more specific annotations for animals, living organisms, humans, etc.. especially in terms of emotional intelligence and alerting the respective rangers to avoid the dispute in the meantime by integrating APIs or faster SMS' or the fastest method to use acknowledgments.

## Conclusion:
This project not only showcases the potential of AI in enhancing safety but also opens avenues for further research and development in emotion recognition technology. By continually refining the model and expanding its capabilities, we can contribute significantly to creating safer environments across various sectors.

## Contributions and Acknowledgments:
- Jahnvi Paliwal
- Biplab Das S (for courses and project support) ![link](https://www.linkedin.com/in/biplab-das-7b9870165/)

