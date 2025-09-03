from flwr.client import NumPyClient, ClientApp
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import toml
from trans import TransformerModelHandler

config = toml.load("pyproject.toml")
epochs = config["training"]["local_epochs"]

def load_data(train_path='data/Client2/train', test_path='data/Client2/test'):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train_dataset = torchvision.datasets.ImageFolder(root=train_path, transform=transform)
    test_dataset = torchvision.datasets.ImageFolder(root=test_path, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    return train_loader, test_loader

# Flower client
class FlowerClient(NumPyClient):
    def __init__(self, model, train_loader, test_loader):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader

    # TransformerModelHandler doesn't use get_parameters/set_parameters in the same way
    def get_parameters(self, config):
        return []

    def set_parameters(self, parameters):
        pass

    def fit(self, parameters, config):
        self.model.train(self.train_loader, epochs)
        # Save model state dictionary if needed
        torch.save(self.model.model.state_dict(), "trained_model_client_2_transformer.pt")
        return [], len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        loss, accuracy = self.model.evaluate(self.test_loader)
        print(f"Client Evaluation - Loss: {loss:.4f}, Accuracy: {accuracy:.2f}")
        return loss, len(self.test_loader.dataset), {"accuracy": accuracy, "loss": loss}

# Create transformer model instance
net = TransformerModelHandler()

def client_fn(cid: str):
    """Create and return an instance of Flower `Client`."""
    train_loader, test_loader = load_data()
    return FlowerClient(net, train_loader, test_loader).to_client()

# Flower ClientApp
app = ClientApp(
    client_fn=client_fn,
)

if __name__ == "__main__":
    from flwr.client import start_client

    train_loader, test_loader = load_data()

    start_client(
        server_address="127.0.0.1:5006",
        client=FlowerClient(net, train_loader, test_loader).to_client(),
    )
