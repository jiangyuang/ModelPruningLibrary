import os
from time import sleep
from timeit import default_timer as timer
from bases.sockets import ClientSocket
from bases.messages import ClientToServerUpdateMessage
from datasets.loader import get_train_loader
from configs.mnist import EXP_NAME, PRUNING_TYPE_NAMES
from configs.constants import MNIST

from configs.network import SERVER_ADDR, SERVER_PORT
from utils.data_iterator import DataIterator
from utils.save_load import mkdir_save


def main():
    client = ClientSocket(SERVER_ADDR, SERVER_PORT)
    init_msg = client.init_connections()

    pruning_type, n_pruning_levels, seed = init_msg.server_params
    model = init_msg.model
    model.train()
    data_indices = init_msg.slice_indices
    n_local_updates = init_msg.n_local_updates
    batch_size = init_msg.batch_size
    pruning_type_name = PRUNING_TYPE_NAMES[pruning_type]

    if pruning_type == 0:
        path = os.path.join("results", EXP_NAME, CLIENT_NAME, pruning_type_name, "seed_" + str(seed))
    else:
        path = os.path.join("results", EXP_NAME, CLIENT_NAME, pruning_type_name, "level_" + str(n_pruning_levels),
                            "seed_" + str(seed))

    train_loader = get_train_loader(EXP_NAME, train_batch_size=MNIST.N_TRAIN, shuffle=False, flatten=True,
                                    train_set_indices=data_indices, one_hot=True, n_workers=16, pin_memory=True)

    train_iter = DataIterator(data_loader=train_loader, batch_size=batch_size)
    list_t_computation, list_t_communication = [], []
    print("CLIENT. PRUNING TYPE = {}, N_PRUNING_LEVELS = {}, SEED = {}.".format(pruning_type_name, n_pruning_levels,
                                                                                seed))

    # Write system status into files
    # status_writer_process = Process(target=_status_writer)
    # status_writer_process.start()

    while True:
        t_start = timer()
        model.zero_grad()
        for _ in range(n_local_updates):
            inputs, labels = train_iter.get_next_batch()
            loss = model.loss(inputs, labels)
            loss.backward()
            model.apply_grad()

        t_comp = timer()

        client.send_msg(ClientToServerUpdateMessage([model.state_dict(), batch_size]))
        update_msg = client.recv_update_msg()
        state_dict = update_msg.state_dict

        model.load_state_dict(state_dict)

        t_end = timer()
        list_t_computation.append(t_comp - t_start)
        list_t_communication.append(t_end - t_comp)

        mkdir_save(list_t_computation, os.path.join(path, "computation_time"))
        mkdir_save(list_t_communication, os.path.join(path, "communication_time"))

        terminate = update_msg.terminate
        if terminate:
            print("Task completed")
            break

    client.close()


if __name__ == "__main__":
    CLIENT_NAME = "client"
    main()
    sleep(5)
