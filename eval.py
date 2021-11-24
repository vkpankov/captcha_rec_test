from model import CaptchaRecModel
from dataset import ImageDataSet
from torchvision import transforms
import utils
import argparse
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate captcha recognizer")

    parser.add_argument(
        "--source",
        metavar="source",
        default="dataset/test",
        type=str,
        help="Input folder with images",
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
        metavar="batchsize",
        default=32,
        type=int,
        help="Batch size",
    )

    args = parser.parse_args()

    tfs = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            transforms.Grayscale(),
        ]
    )
    data = ImageDataSet(args.source, transform=tfs)
    data_loader = torch.utils.data.DataLoader(
        data, batch_size=args.batch_size, shuffle=True
    )

    model = CaptchaRecModel().to(args.device)
    model.load_state_dict(torch.load("last_model.pkl"))
    model.eval()
    correct = 0
    n = 0
    for imgs, labels in data_loader:
        pred = model(imgs.to(args.device))
        pred = pred.to("cpu").detach().numpy()
        for p, l in zip(pred, labels):
            pred_str = utils.onehot_decode(p)
            pred_l = utils.onehot_decode(l)
            if pred_str == pred_l:
                correct += 1
            n += 1
    print(f"Accuracy: {1.0 * correct / n}, {correct}/{n}")
