from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from data import DATASET_DIR
from data.PokerDataset import STATE_TARGET, data_loader_factory
from enums.GameType import GameType
from enums.StateTargetType import StateTargetType
from networks.StateDetector import StateDetector
import torch.nn as nn

DATASET_NAME = '9_player_state'
BATCH_SIZE = 5
EPOCHS = 50
LR = 0.001
SAVE_NAME = '9_player_state_detector'
GAME_TYPE = GameType.NinePlayer


def run(model, dataloader, device, is_train, debug):
    total_loss = 0
    total_correct_player = 0
    total_correct_table = 0
    total_correct_dealer = 0
    total_correct_opponents = 0
    total_samples = 0

    with tqdm(dataloader, unit="batch") as tepoch:
        for x, y in tepoch:
            tepoch.set_description("train" if is_train else "test")
            x = x.to(device)
            y.to(device)

            player_cards, table_cards, dealer_pos, opponents = model.forward(
                x)

            loss_func = nn.CrossEntropyLoss()

            player_target, _ = y[StateTargetType.NumPlayerCards]
            player_loss = loss_func(player_cards, player_target)

            table_target, uuids = y[StateTargetType.NumTableCards]
            table_loss = loss_func(table_cards, table_target)

            dealer_target, _ = y[StateTargetType.DealerPosition]
            dealer_loss = loss_func(dealer_pos, dealer_target)

            opponent_targets, _ = y[StateTargetType.Opponents]
            opponent_loss_fc = nn.BCEWithLogitsLoss()
            opponent_loss = opponent_loss_fc(
                opponents, opponent_targets.to(torch.float32))

            batch_loss = player_loss + table_loss + \
                dealer_loss + opponent_loss

            if is_train:
                model.optim.zero_grad()
                batch_loss.backward()
                model.optim.step()

            with torch.no_grad():
                softmax = nn.Softmax(dim=1)
                sigmoid = nn.Sigmoid()
                num_correct_player = (
                    torch.argmax(softmax(player_cards), 1) == player_target).long().sum()
                num_correct_target = (
                    torch.argmax(softmax(table_cards), 1) == table_target).long().sum()
                num_correct_dealer = (
                    torch.argmax(softmax(dealer_pos), 1) == dealer_target).long().sum()

                num_correct_opponent = (torch.flatten(
                    sigmoid(opponents) > 0.5) == torch.flatten(opponent_targets)).long().sum()

                total_correct_player += num_correct_player.item()
                total_correct_table += num_correct_target.item()
                total_correct_dealer += num_correct_dealer.item()
                total_correct_opponents += num_correct_opponent.item()

                total_loss += batch_loss.item()
                total_samples += len(x)

                if debug:
                    indicies = (torch.argmax(softmax(table_cards), 1)
                                == table_target)
                    uuids = np.array(uuids)
                    print(uuids[indicies.detach().cpu().numpy()])

    print(f"Total_loss: {total_loss}")
    print(f"Player_accuracy: {total_correct_player*100/total_samples}")
    print(f"Table_accuracy: {total_correct_table*100/total_samples}")
    print(f"Dealer_accuracy: {total_correct_dealer*100/total_samples}")
    print(
        f"Opponent_accuracy: {total_correct_opponents*100/(GAME_TYPE.get_num_players()*total_samples)}")
    print("\n\n")


if __name__ == "__main__":
    dataset_dir = Path(f"{DATASET_DIR}/{DATASET_NAME}")
    train_loader, test_loader = data_loader_factory(
        dataset_dir, 0.7, batch_size=BATCH_SIZE, target_type=STATE_TARGET)

    num_train = len(train_loader.dataset)
    num_test = len(test_loader.dataset)
    print(f"{num_train} train samples, {num_test} test samples")

    device = "cuda" if torch.cuda.is_available() else 'cpu'
    print(f"Training on {device} for {EPOCHS}")

    network = StateDetector(lr=LR, game_type=GAME_TYPE)
    network.to(device)

    for e in range(EPOCHS):
        print(f"\nEpoch {e+1}")
        network.train()
        run(network, train_loader, device, is_train=True, debug=False)

        network.eval()
        with torch.no_grad():
            run(network, test_loader,  device,
                is_train=False, debug=False)

    network.save(SAVE_NAME)
