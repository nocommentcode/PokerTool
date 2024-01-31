from pathlib import Path
import torch
from tqdm import tqdm
from data.PokerDataset import data_loader_factory
from enums.TargetType import TargetType
from networks.CardDetector import CardDetector
import torch.nn as nn


def get_target_counts(batch_target):
    def get_card_count(target_type):
        suit, _ = batch_target[target_type]
        return (suit != 0)

    target_counts = torch.zeros((len(batch_target), 2)).to("cuda")

    for target in [TargetType.Player_card_1, TargetType.Player_card_2]:
        target_counts[:, 0] += get_card_count(target)

    for target in [TargetType.Flop_card_1, TargetType.Flop_card_2, TargetType.Flop_card_3, TargetType.Turn_card, TargetType.River_card]:
        target_counts[:, 1] += get_card_count(target)

    return target_counts


def run(model, dataloader, device, is_train):
    total_loss = 0
    total_correct_player = 0
    total_correct_table = 0
    total_samples = 0
    with tqdm(dataloader, unit="batch") as tepoch:
        for x, y in tepoch:
            tepoch.set_description("train" if is_train else "test")
            x = x.to(device)
            y.to(device)

            player_cards, table_cards = model.forward(x)
            target_counts = get_target_counts(y)

            loss_func = nn.CrossEntropyLoss()
            player_loss = loss_func(player_cards, target_counts[:, 0].long())
            table_loss = loss_func(table_cards, target_counts[:, 1].long())
            batch_loss = player_loss + table_loss

            if is_train:
                model.optim.zero_grad()
                batch_loss.backward()
                model.optim.step()

            with torch.no_grad():
                softmax = nn.Softmax(dim=1)
                num_correct_player = (
                    torch.argmax(softmax(player_cards), 1) == target_counts[:, 0]).long().sum()
                num_correct_target = (
                    torch.argmax(softmax(table_cards), 1) == target_counts[:, 1]).long().sum()

                total_correct_player += num_correct_player.item()
                total_correct_table += num_correct_target.item()

                total_loss += batch_loss.item()

                total_samples += len(x)

    print(f"Total_loss: {total_loss}")
    print(f"Player_accuracy: {total_correct_player*100/total_samples}")
    print(f"Table_accuracy: {total_correct_table*100/total_samples}")
    print("\n\n")


if __name__ == "__main__":
    dataset_dir = Path("images/classified_images")
    train_loader, test_loader = data_loader_factory(
        dataset_dir, 0.7, batch_size=32)

    num_train = len(train_loader.dataset)
    num_test = len(test_loader.dataset)
    print(f"{num_train} train samples, {num_test} test samples")
    epochs = 10
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    print(f"Training on {device} for {epochs}")

    network = CardDetector(lr=0.001)
    network.to(device)

    for e in range(epochs):
        print(f"\nEpoch {e+1}")
        network.train()
        run(network, train_loader, device, is_train=True)

        network.eval()
        with torch.no_grad():
            run(network, test_loader,  device, is_train=False)

    network.save("card_detector")
