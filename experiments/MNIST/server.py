import os
import argparse
from timeit import default_timer as timer
import torch
from threading import Thread

from bases.sockets import ServerSocket
from bases.messages import ServerToClientInitMessage, ServerToClientUpdateMessage
from bases.nn.models import MNISTModel
from configs.mnist import *
from configs.constants import MNIST
from configs.network import SERVER_ADDR, SERVER_PORT
from datasets.loader import get_test_loader
from utils.functional import random_split
from utils.save_load import mkdir_save
from utils.data_iterator import DataIterator


def load_model():
    model = MNISTModel(lr=LR)
    if PRUNING_TYPE == 0:
        return model
    elif PRUNING_TYPE == 1:
        for _ in range(N_PRUNING_LEVELS):
            model.prune_by_pct("classifier", PRUNING_PCT)
        return model.to_sparse()
    elif PRUNING_TYPE == 2:
        for _ in range(N_PRUNING_LEVELS):
            model.random_prune_by_pct("classifier", PRUNING_PCT)
        return model.to_sparse()
    else:
        raise RuntimeError("Pruning type {} is not supported".format(PRUNING_TYPE))


def eval_async(func, iterator, loss_list: list, acc_list: list):
    l, a = func(iterator)
    loss_list.append(l)
    acc_list.append(a)
    mkdir_save(loss_list, os.path.join(DATA_DIR_PATH, "loss"))
    mkdir_save(acc_list, os.path.join(DATA_DIR_PATH, "accuracy"))


def main():
    # Wait for connections
    server = ServerSocket(SERVER_ADDR, SERVER_PORT, N_CLIENTS)
    server.wait_for_connections()
    torch.manual_seed(SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Current device is {}.".format(device))

    list_loss, list_acc, list_time = [], [], []
    model = load_model()
    random_indices = random_split(MNIST.N_TRAIN, N_CLIENTS)
    test_loader = get_test_loader(EXP_NAME, test_batch_size=MNIST.N_TEST, shuffle=False, flatten=True,
                                  one_hot=True, n_workers=16, pin_memory=True)
    test_iter = DataIterator(data_loader=test_loader, batch_size=200, device=device)

    init_msgs = [ServerToClientInitMessage(
        [model, torch.tensor(random_indices[idx]), N_LOCAL_UPDATES, CLIENT_BATCH_SIZE,
         (PRUNING_TYPE, N_PRUNING_LEVELS, SEED)]) for idx in range(N_CLIENTS)]

    server.init_connections(init_msgs)
    model.eval()

    print("SERVER. PRUNING TYPE = {}, N_PRUNING_LEVELS = {}, PRUNING_PCT = {}, SEED = {}, N_ITERATIONS = {}.".format(
        PRUNING_TYPE_NAME, N_PRUNING_LEVELS, PRUNING_PCT, SEED, N_ITERATIONS))

    prev_thread = None
    t_start = timer()
    for idx in range(N_ITERATIONS):
        t_fed_start = timer()
        msgs = server.recv_update_msg_from_all()

        list_state_dict = [msg.state_dict for msg in msgs]
        avg_state_dict = model.state_dict().copy()
        for key in avg_state_dict.keys():
            new_val = None
            for state_dict in list_state_dict:
                if new_val is None:
                    new_val = state_dict[key]
                else:
                    new_val += state_dict[key]
            new_val /= N_CLIENTS
            avg_state_dict[key] = new_val

        model.load_state_dict(avg_state_dict)

        if idx % EVAL_INTERVAL == 0:
            # Asynchronously evaluate model
            if prev_thread is not None:
                prev_thread.join()
            t = Thread(target=eval_async, args=(model.evaluate, test_iter, list_loss, list_acc))
            t.start()
            prev_thread = t

        if idx % DISP_SAVE_INTERVAL == 0:
            print("Federation #{}".format(idx))
            if len(list_loss) != 0 and len(list_acc) != 0:
                loss, acc = list_loss[-1], list_acc[-1]
                print("Loss/acc at iteration {} = {}/{}".format((len(list_loss) - 1) * EVAL_INTERVAL, loss, acc))
                t = timer()
                print("Elapsed time = {}".format(t - t_start))

        terminate = True if idx == N_ITERATIONS - 1 or timer() - t_start >= MAX_TIME else False

        server.send_msg_to_all(ServerToClientUpdateMessage([model.state_dict(), terminate]))
        t_fed_end = timer()
        list_time.append(t_fed_end - t_fed_start)
        if terminate:
            prev_thread.join()

        # Saving loss/acc
        mkdir_save(list_loss, os.path.join(DATA_DIR_PATH, "loss"))
        mkdir_save(list_acc, os.path.join(DATA_DIR_PATH, "accuracy"))
        mkdir_save(list_time, os.path.join(DATA_DIR_PATH, "time"))
        mkdir_save(model, os.path.join(DATA_DIR_PATH, "model"))

        if terminate:
            break

    print("Task completed")
    server.close()


def _parse_args():
    parser = argparse.ArgumentParser(description="Parse command line arguments")
    parser.add_argument('-t', '--type',
                        help="Pruning types: 1 for initialization based, 2 for random, (default = no pruning)",
                        action='store',
                        dest='PRUNING_TYPE',
                        type=int,
                        default=0,
                        nargs='?',
                        required=False)
    parser.add_argument('-l', '--level',
                        help="Number of pruning levels",
                        action='store',
                        dest='N_PRUNING_LEVELS',
                        type=int,
                        nargs='?',
                        required=False)
    parser.add_argument('-sd', '--seed',
                        help="Seed",
                        action='store',
                        dest='SEED',
                        type=int,
                        default=0,
                        nargs='?',
                        required=False)

    return vars(parser.parse_args())


if __name__ == "__main__":
    args = _parse_args()
    PRUNING_TYPE, N_PRUNING_LEVELS, SEED = args["PRUNING_TYPE"], args["N_PRUNING_LEVELS"], args["SEED"]
    if (PRUNING_TYPE == 1 or PRUNING_TYPE == 2) and N_PRUNING_LEVELS is None:
        raise RuntimeError("Level argument must be provided.")

    PRUNING_TYPE_NAME = PRUNING_TYPE_NAMES[PRUNING_TYPE]
    N_ITERATIONS = N_FEDERATIONS
    SERVER_NAME = "server"

    if PRUNING_TYPE == 0:
        DATA_DIR_PATH = os.path.join("results", EXP_NAME, SERVER_NAME, PRUNING_TYPE_NAME, "seed_" + str(SEED))
    else:
        DATA_DIR_PATH = os.path.join("results", EXP_NAME, SERVER_NAME, PRUNING_TYPE_NAME,
                                     "level_" + str(N_PRUNING_LEVELS), "seed_" + str(SEED))
    main()
