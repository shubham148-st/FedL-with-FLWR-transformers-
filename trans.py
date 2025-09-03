import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

class TransformerModelHandler:
    def __init__(self, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model and processor
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32",use_fast=False)
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
    
    def prepare_inputs(self, images, texts):
        # images: list of PIL images, texts: list of strings
        inputs = self.processor(images=images, text=texts, return_tensors="pt", padding=True).to(self.device)
        return inputs
    
    def train(self, train_loader, epochs=3):
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)
        criterion = torch.nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            for batch in train_loader:
                images, labels = batch
                pil_images = []
                for img in images:
                    if torch.is_tensor(img):
                        arr = img.permute(1, 2, 0).numpy()
                        if arr.dtype == 'float32' or arr.dtype == 'float64':
                            arr = (arr * 255).astype('uint8')
                        pil_img = Image.fromarray(arr)
                        pil_images.append(pil_img)
                    else:
                        pil_images.append(img)                
                texts = ["a photo of Mahua disease" if l == 1 else "a photo of Non Disease mahua plant" for l in labels]
                inputs = self.prepare_inputs(pil_images, texts)
                outputs = self.model(**inputs)
                logits = outputs.logits_per_image
                loss = criterion(logits, labels.to(self.device))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
    def evaluate(self, test_loader):
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        criterion = torch.nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for images, labels in test_loader:
                pil_images = []
                for img in images:
                    if torch.is_tensor(img):
                        arr = img.permute(1, 2, 0).numpy()
                        if arr.dtype == 'float32' or arr.dtype == 'float64':
                            arr = (arr * 255).astype('uint8')
                        pil_img = Image.fromarray(arr)
                        pil_images.append(pil_img)
                    else:
                        pil_images.append(img)
                texts = ["a photo of Mahua disease" if l == 1 else "a photo of Non Disease mahua plant" for l in labels]
                inputs = self.prepare_inputs(pil_images, texts)
                outputs = self.model(**inputs)
                logits = outputs.logits_per_image
                loss = criterion(logits, labels.to(self.device))
                
                total_loss += loss.item() * labels.size(0)
                preds = torch.argmax(logits, dim=1)
                total_correct += (preds == labels.to(self.device)).sum().item()
                total_samples += labels.size(0)
                
        avg_loss = total_loss / total_samples
        accuracy = (total_correct / total_samples) *100
        return avg_loss, accuracy
