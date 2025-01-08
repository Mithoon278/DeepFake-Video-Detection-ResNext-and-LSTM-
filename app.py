from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import time
import torch
import cv2
import numpy as np
from torch import nn
from torchvision import models
from torchvision import transforms

# Flask app initialization
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'gif', 'webm', 'avi', '3gp', 'wmv', 'flv', 'mkv'}
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # Max file size: 100 MB

# Model and transform setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None  # Placeholder for loading the saved model

im_size = 112
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
sm = nn.Softmax(dim=1)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((im_size, im_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def predict_frame(model, frame):
    """
    Preprocess a single frame and run it through the model.
    """
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(img_tensor)
        logits = sm(logits)
        _, prediction = torch.max(logits, 1)
        confidence = logits[0][prediction].item() * 100
    return prediction.item(), confidence

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = f"uploaded_{int(time.time())}.{file.filename.rsplit('.', 1)[1].lower()}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        return jsonify({'message': 'File uploaded successfully', 'filename': filename}), 200
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/predict', methods=['POST'])
def predict_video():
    data = request.json
    filename = data.get('filename')
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    if not os.path.exists(video_path):
        return jsonify({'error': 'File not found'}), 404

    # Process the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return jsonify({'error': 'Unable to process video'}), 500

    predictions = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % 10 == 0:  # Predict every 10th frame for efficiency
            pred, confidence = predict_frame(model, frame)
            predictions.append({'frame': frame_count, 'prediction': pred, 'confidence': confidence})

    cap.release()

    return jsonify({'predictions': predictions}), 200

@app.before_first_request
def load_model():
    global model

    # Define the model architecture
    class Model(nn.Module):
        def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
            super(Model, self).__init__()
            model = models.resnext50_32x4d(pretrained=True)  # Residual Network CNN
            self.model = nn.Sequential(*list(model.children())[:-2])  # Remove the final layers
            self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
            self.relu = nn.LeakyReLU()
            self.dp = nn.Dropout(0.4)
            self.linear1 = nn.Linear(2048, num_classes)
            self.avgpool = nn.AdaptiveAvgPool2d(1)

        def forward(self, x):
            batch_size, seq_length, c, h, w = x.shape
            x = x.view(batch_size * seq_length, c, h, w)
            fmap = self.model(x)  # Feature map from ResNext
            x = self.avgpool(fmap)
            x = x.view(batch_size, seq_length, 2048)
            x_lstm, _ = self.lstm(x, None)  # LSTM for temporal information
            return fmap, self.dp(self.linear1(torch.mean(x_lstm, dim=1)))

    # Initialize the model with the correct number of classes
    model = Model(num_classes=2).cuda()

    # Load the state_dict into the model
    model.load_state_dict(torch.load('saved_model.pt', map_location='cuda'))
    model.eval()  # Set model to evaluation mode

    print("Model loaded successfully.")


# Static files (e.g., for videos)
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Run Flask app
if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
