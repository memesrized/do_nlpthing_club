import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dnc.distillation.code.dataset import EmbDataset
from dnc.distillation.code.log import get_logger
from dnc.distillation.code.model import EmbClassifier
from dnc.distillation.code.tokenizer import Tokenizer

logger = get_logger(__name__)


def train(
    model,
    dataloader,
    device,
    optimizer,
    criterion,
    n_epochs=10,
    lr=0.00001,
    clip=5,
    model_folder="",
):
    data_len = len(dataloader)

    model.train()

    loss_array = []

    for epoch in range(1, n_epochs + 1):
        for i, batch in tqdm(enumerate(dataloader), total=data_len):
            optimizer.zero_grad()
            output = model(batch[0].to(device))
            output = output.to(device)
            target_seq = batch[1].to(device)
            loss = criterion(output, target_seq)
            loss_array.append(loss.item())
            loss.backward()
            # TODO: google about usage
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            # scheduler.step()
            if i % 400 == 0:
                print(
                    "Batch_100: {}/{}.............".format(
                        int(i / 400), int(data_len / 400) + 1
                    ),
                    end=" ",
                )
                print("Loss: {:.4f}".format(loss.item()))
        model.save(f"{model_folder}models{epoch}")

    return model, loss_array


if __name__ == "__main__":
    EMB_PATH = "navec_hudlit_v1_12B_500K_300d_100q.tar"
    DF_PATH = "train.csv"

    tokenizer = Tokenizer(EMB_PATH)
    dataset = EmbDataset("train.csv", tokenizer, size=50000)

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.cuda.empty_cache()

    if device == "cuda":
        print(torch.cuda.get_device_name(0))
        print("Memory Usage:")
        print("Allocated:", round(torch.cuda.memory_allocated(0) / 1024**3, 1), "GB")
        print("Cached:   ", round(torch.cuda.memory_reserved(0) / 1024**3, 1), "GB")

    model = EmbClassifier(emb_dim=300, hid_dim=512)
    model = model.to(device)

    n_epochs = 10
    lr = 0.00001
    clip = 5

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=lr)

    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr,
    #                                                 steps_per_epoch=len(dataloader), epochs=n_epochs)

    data_len = len(dataloader)

    model.train()

    loss_array = []

    for epoch in range(1, n_epochs + 1):
        for i, batch in tqdm(enumerate(dataloader), total=data_len):
            optimizer.zero_grad()
            output = model(batch[0].to(device))
            output = output.to(device)
            target_seq = batch[1].to(device)
            #         print(target_seq.view(-1).long().shape)
            #         print(output.shape)
            loss = criterion(output, target_seq)
            loss_array.append(loss.item())
            loss.backward()
            # TODO: google about usage
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            # scheduler.step()
            if i % 400 == 0:
                print(
                    "Batch_100: {}/{}.............".format(
                        int(i / 400), int(data_len / 400) + 1
                    ),
                    end=" ",
                )
                print("Loss: {:.4f}".format(loss.item()))
        model.save(f"models{epoch}")

    loss_array_smaller = [x for i, x in enumerate(loss_array) if i % 400 == 0]

    fig, ax = plt.subplots()
    ax.plot(range(len(loss_array_smaller)), loss_array_smaller)

    ax.set(xlabel="batch", ylabel="loss", title="Train loss function")
    ax.grid()

    fig.savefig("test_512_10_epochs_50k.png")
    plt.show()
