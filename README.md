**Web App** - https://rahil-audio-mnist-classification.streamlit.app
**Audio MNIST Classification**

This project implements an audio classification model using Streamlit, PyTorch, and CNN to distinguish between spoken digits (0-9) based on the AudioMNIST dataset.

**Key Features:**

- Streamlit app for user-friendly interaction with the model.
- PyTorch for efficient deep learning model development.
- Convolutional Neural Network (CNN) architecture tailored for audio classification.

**Getting Started**

1. **Prerequisites:**
   - Python 3.9 ([https://www.python.org/downloads/](https://www.python.org/downloads/))
   - pip package manager (usually included with Python)
   - Git version control system ([https://git-scm.com/](https://git-scm.com/))

2. **Clone the Repository:**
   ```bash
   git clone https://github.com/rahilfaizan/Audio-Mnist-Classification.git
   cd Audio-Mnist-Classification
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the App:**
   ```bash
   streamlit run app.py
   ```

   This will launch the Streamlit app in your web browser, typically at `http://localhost:8501`.

**Project Structure:**

```
rahil-audio-mnist-classification/
├── app.py                     # Streamlit app for user interaction
├── predict.py                 # PyTorch CNN model prediction
├── convert_audio.py           # functions for data preprocessing, etc.
└── requirements.txt           # List of required Python packages
```

**How it Works:**

1. The `app.py` script defines the Streamlit app interface.
2. Users can upload an audio file of a spoken digit.
3. The app preprocesses the audio using functions from `convert_audio.py`.
4. The preprocessed audio is fed into the CNN model defined in `predict.py`.
5. The model predicts the most likely digit (0-9).
6. The predicted digit is displayed in the Streamlit app.

**Customization:**

- Modify the CNN architecture in `predict.py` to experiment with different network configurations.
- Adjust data preprocessing steps in `convert_audio.py` to explore alternative techniques.

**Deployment:**

- To deploy the Streamlit app to a production environment, consider services like Heroku, Streamlit Cloud(what I chose), or Google Cloud Run.

**Additional Notes:**
- I've employed a pre-trained model for this application, which I trained independently.
- If you want to build your own model you'll need to download the AudioMNIST dataset manually and train it.
