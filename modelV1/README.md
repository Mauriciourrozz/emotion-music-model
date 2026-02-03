## Training the model

This project does not include a pre-trained model.
The emotion classifier must be trained locally.

Steps:

1. Install dependencies
   pip install -r requirements.txt

2. Process raw data (GO Emotions → Ekman)
   python data_processed.py

3. Tokenize the data
   python tokenize_data.py

4. Train the model
   python model_training.py

5. Run emotion prediction
   python predict_emotion.py
