import torch as t
from torchvision import datasets, transforms

# Load MNIST data
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST("./data", train=False, download=True, transform=transform)


def get_filtered_loader(training=True, max_label=1):
    dataset = train_dataset if training else test_dataset
    idx = [i for i, (_, label) in enumerate(dataset) if label <= max_label]
    if training:
        idx = idx[:12665]  # Size for 0-1 subset (for consistency)
    else:
        idx = idx[:2115]  # Size for 0-1 subset (for consistency)
    filtered_dataset = t.utils.data.Subset(dataset, idx)
    return (
        t.utils.data.DataLoader(filtered_dataset, batch_size=256, shuffle=True),
        filtered_dataset,
  )
