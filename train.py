import torch
from torchvision import transforms
from torch.optim.adam import Adam
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataset import Dataset
from torchvision.transforms.autoaugment import AutoAugmentPolicy
import utils
from dataset import ImageDataSet
from model import CaptchaRecModel
import os
import argparse

b1 = 0.6
b2 = 0.9

parser = argparse.ArgumentParser(description="Train captcha recognizer")

parser.add_argument(
    "--train_folder",
    metavar="train_folder",
    default="dataset/train",
    type=str,
    help="Train images folder",
)

parser.add_argument(
    "--test_folder",
    metavar="test_folder",
    default="dataset/test",
    type=str,
    help="Test images folder",
)


parser.add_argument(
    "--learning_rate",
    metavar="learning_rate",
    default=0.0005,
    type=float,
    help="Learning rate",
)
parser.add_argument(
    "--device",
    metavar="device",
    default="cuda",
    type=str,
    help="Device: cpu or cuda",
)

parser.add_argument(
    "--batch_size",
    metavar="batch_size",
    default=32,
    type=int,
    help="Batch size",
)

parser.add_argument(
    "--epochs_count",
    metavar="epochs_count",
    default=100,
    type=int,
    help="Epochs count",
)

parser.add_argument(
    "--saved_model",
    metavar="saved_model",
    default="last_model.pkl",
    type=str,
    help="Model checkpoint",
)


args = parser.parse_args()


def train(model: CaptchaRecModel, opt: Optimizer, data: Dataset):
    criterion = torch.nn.BCELoss()
    n = 0
    model.train()
    for img, label in data:
        opt.zero_grad()
        pred = model(img.to(args.device))
        loss = criterion(pred, label.to(args.device))
        loss.backward()
        opt.step()
        n += 1
        if n % 5 == 0:
            print(f"#{n}/{len(data)}, loss: {loss}")


def validate(model: CaptchaRecModel, data: Dataset):
    model.eval()
    correct = 0
    n = 0
    for imgs, labels in data:
        pred = model(imgs.to(args.device))
        pred = pred.to("cpu").detach().numpy()
        for p, l in zip(pred, labels):
            pred_str = utils.onehot_decode(p)
            pred_l = utils.onehot_decode(l)
            if pred_str == pred_l:
                correct += 1
            n += 1
    return 1.0 * correct / n


if __name__ == "__main__":

    tfs = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            transforms.Grayscale(),
        ]
    )

    train_dataset = ImageDataSet(args.train_folder, transform=tfs)
    test_dataset = ImageDataSet(args.test_folder, transform=tfs)
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=True
    )

    model = CaptchaRecModel().to(args.device)
    if os.path.exists(args.saved_model):
        model.load_state_dict(torch.load(args.saved_model))
    opt = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, betas=(b1, b2)
    )

for i in range(0, args.epochs_count):
    print(f"##### Epoch: {i} #####")
    train(model, opt, train_data_loader)
    test_val = validate(model, test_data_loader)
    print(f"Test accuracy: {test_val}")
    torch.save(model.state_dict(), "last_model2.pkl")
