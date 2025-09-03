🧠 FedL-with-Transformers-ViT 🚀
This project implements a Federated Learning (FL) system for image-based plant disease detection using PyTorch, Vision Transformer (ViT) models, and the Flower (FLWR) framework. The model is trained across multiple clients without sharing raw data, ensuring data privacy.

⚠️ Note
This code is adapted for Vision Transformers (ViT).

Supports small-scale university-level AIML projects (4–5 clients, broadcasted network).
But I used only two clients for this very project.
Upgrade code for the latest Flower versions (e.g., 1.x) if required.

📦 Dependencies
Install these Python libraries:

text
flwr==1.20.0
torch==2.8.0
torchvision==0.23.0
transformers==4.56.0
scikit-learn==1.7.1
numpy==2.3.2
matplotlib==3.10.6
grpcio==1.74.0
requests==2.32.5
tqdm==4.67.1

Install with:


📁 Data Setup
Organize your dataset per client:

text
Client_1/
├── train/
└── test/
Client_2/
├── train/
└── test/
Label images as:

0 → Healthy

1 → Diseased

Update class labels in trans.py if modified.

💪 Using Pre-trained Models
Use provided pre-trained transformer models: trained_model_client1_transformers.pt, etc.

(Dataset: I used approx 3k images and both the client are using same dataset.)


⚙️ Running Federated Training from Scratch
Install dependencies (preferably in a virtual environment).

Open three terminals and run:

Terminal 1:

text
python server.py
Terminal 2:

text
python client_111.py
Terminal 3:

text
python client_112.py
The system will start ViT-based federated training using FedAvg.

Accuracy/loss graphs will be shown.

Logs (in CSV) and trained ViT model weights for each client will be saved.

📝 Notes
Designed for Federated Learning experiments (ViT) via Flower.

Update script paths if the folder structure changes.

Virtual environments recommended for smooth package management.

For issues or improvements, please contribute or submit a pull request.
