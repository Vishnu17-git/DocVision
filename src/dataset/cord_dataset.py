try:
    from torch.utils.data import Dataset
except ModuleNotFoundError:  # pragma: no cover - fallback for environments without torch
    class Dataset:  # type: ignore[no-redef]
        """Fallback Dataset base when PyTorch is unavailable."""

        pass


class CORDDataset(Dataset):

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        sample = self.dataset[idx]

        image = sample["image"]
        ground_truth = sample["ground_truth"]

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "ground_truth": ground_truth
        }