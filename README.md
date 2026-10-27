# Facial Emotion Recognition – FER2013

## Overview
A deep learning application that uses a Convolutional Neural Network (CNN) to recognize facial emotions from images. The model was trained on the FER2013 dataset using Keras and deployed with Streamlit.

## Live App
[Try the Facial Emotion Recognition App](https://emotion-detection-fer2013.streamlit.app/)

## Features
- Upload an image or capture a photo using a webcam.
- Predict facial emotions using a trained CNN model.
- Classify faces into 7 emotions: Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise.
- Display prediction confidence and the probability of each emotion.
- Visualize emotion probabilities using a bar chart.

## Technologies Used
- Python
- Keras / TensorFlow
- Streamlit
- NumPy and Pandas
- Matplotlib and Seaborn
- Scikit-learn

## Model
- **Dataset:** FER2013
- **Model:** Convolutional Neural Network (CNN)
- **Input:** 48×48 grayscale facial images
- **Output:** 7 emotion classes
- **Techniques:** Data Augmentation, Batch Normalization, Dropout, and Max Pooling

## Limitations
- Prediction accuracy depends on image quality, lighting, and facial positioning.
- Similar facial expressions may be difficult to distinguish.
- Class imbalance in FER2013 can affect predictions, especially for underrepresented emotions.
- Predictions represent model estimates and may not reflect a person's actual emotional state.

## Future Improvements
- Improve performance on underrepresented emotions.
- Add automatic face detection and cropping.
- Explore transfer learning to improve classification accuracy.
