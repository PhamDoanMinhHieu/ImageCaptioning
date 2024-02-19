import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from utils import save_checkpoint, load_checkpoint, print_examples
from get_loader import get_loader
from model import CNNtoRNN


def train():
    # Khởi tạo transform
    transform = transforms.Compose(
        [
            transforms.Resize((356, 356)), # Thay đổi kích thước hình ảnh thành (356, 356) pixels.
            transforms.RandomCrop((299, 299)), # Cắt ngẫu nhiên một vùng hình ảnh có kích thước.
            transforms.ToTensor(), # Chuyển đổi hình ảnh thành tensor PyTorch.
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # Chuyển đổi hình ảnh thành tensor PyTorch.
        ]
    )

    # Khởi tạo dataloader
    train_loader, dataset = get_loader(
        root_folder="./data/flickr8k/images/",
        annotation_file="./data/flickr8k/captions.txt",
        transform=transform,
        num_workers=2,
    )

    # Dòng này chỉ định rằng PyTorch nên sử dụng chế độ tối ưu hóa cuDNN benchmark.
    # Chế độ này cho phép PyTorch tự động tùy chỉnh cấu hình cuDNN để tối ưu hóa hiệu suất của các phép tính trên GPU.
    # Bằng cách bật chế độ benchmark, bạn có thể tận dụng tối đa khả năng tính toán của GPU trong quá trình huấn luyện.
    torch.backends.cudnn.benchmark = True
    
    # Dòng này xác định thiết bị mà PyTorch sẽ sử dụng để tính toán.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 
    load_model = False
    save_model = False
    train_CNN = False

    # Hyperparameters
    embed_size = 256
    hidden_size = 256
    vocab_size = len(dataset.vocab)
    num_layers = 1
    learning_rate = 3e-4
    num_epochs = 100

    # sử dụng để ghi các thông tin huấn luyện và kiểm tra của mô hình vào TensorBoard.
    writer = SummaryWriter("runs/flickr")
    step = 0

    # Khởi tạo model, loss
    model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Only finetune the CNN
    for name, param in model.encoderCNN.inception.named_parameters():
        if "fc.weight" in name or "fc.bias" in name:
            param.requires_grad = True
        else:
            param.requires_grad = train_CNN

    # Nếu model có sẵn thì load model
    if load_model:
        step = load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

    # Chuyển sang chế độ train
    model.train()

    for epoch in range(num_epochs):
        # Uncomment the line below to see a couple of test cases
        # print_examples(model, device, dataset)

        if save_model:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
            }
            save_checkpoint(checkpoint)

        for idx, (imgs, captions) in tqdm(
            enumerate(train_loader), total=len(train_loader), leave=False
        ):
            imgs = imgs.to(device)
            captions = captions.to(device)

            outputs = model(imgs, captions[:-1])
            loss = criterion(
                outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1)
            )

            writer.add_scalar("Training loss", loss.item(), global_step=step)
            step += 1

            optimizer.zero_grad()
            loss.backward(loss)
            optimizer.step()


if __name__ == "__main__":
    train()