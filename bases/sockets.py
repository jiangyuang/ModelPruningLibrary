import socket
from threading import Thread
from time import sleep

import utils.messaging
from bases.messages import MessageTypes, ClientToServerAckMessage


class ClientSocket(socket.socket):
    def __init__(self, server_addr, server_port):
        super(ClientSocket, self).__init__()
        self.server_addr = server_addr
        self.server_port = server_port

    def init_connections(self):
        self.connect_to_server()
        print("Connected to server.")
        init_msg = self.recv_init_msg()
        self.send_msg(ClientToServerAckMessage())
        return init_msg

    def connect_to_server(self):
        for _ in range(100):
            try:
                self.connect((self.server_addr, self.server_port))
                break
            except ConnectionRefusedError:
                sleep(1)
        else:
            raise ConnectionRefusedError("Connection refused")


    def recv_msg(self, expected_msg_type: MessageTypes):
        msg = utils.messaging.recv_msg(self)
        if msg.msg_type != expected_msg_type:
            raise ValueError("Message type should be ", expected_msg_type)
        return msg

    def recv_init_msg(self):
        return self.recv_msg(MessageTypes.ServerToClientInitMessage)

    def recv_update_msg(self):
        return self.recv_msg(MessageTypes.ServerToClientUpdateMessage)

    def send_msg(self, msg):
        utils.messaging.send_msg(self, msg)


class ServerSocket(socket.socket):
    def __init__(self, server_addr, server_port, n_clients):
        super(ServerSocket, self).__init__(socket.AF_INET, socket.SOCK_STREAM)
        self.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.bind((server_addr, server_port))

        self.n_clients = n_clients
        self.list_client_sockets = []

    def init_connections(self, init_msg):
        self.send_msg_to_all(init_msg)
        self.recv_ack_msg_from_all()

    def wait_for_connections(self):
        while len(self.list_client_sockets) < self.n_clients:
            self.listen(self.n_clients * 2)
            print("Waiting for connections... ({}/{})".format(len(self.list_client_sockets), self.n_clients))
            (client_sock, (ip, port)) = self.accept()
            self.list_client_sockets.append(client_sock)
            print('New connection from {}:{}, ({}/{})'.format(ip, port, len(self.list_client_sockets), self.n_clients))

    def recv_msg_from_all(self, expected_msg_type: MessageTypes):
        msgs = [None for _ in range(self.n_clients)]
        threads = []
        for idx in range(self.n_clients):
            t = Thread(target=utils.messaging.recv_msg_async, args=(self.list_client_sockets[idx], msgs, idx))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        for msg in msgs:
            if msg is None:
                raise RuntimeError("Message incomplete")
            elif msg.msg_type != expected_msg_type:
                raise ValueError("Message type should be ", expected_msg_type)

        return msgs

    def recv_ack_msg_from_all(self):
        return self.recv_msg_from_all(MessageTypes.ClientToServerAckMessage)

    def recv_update_msg_from_all(self):
        return self.recv_msg_from_all(MessageTypes.ClientToServerUpdateMessage)

    def send_msg_to_all(self, msgs):
        """
        Supports both msg and list of messages
        """
        threads = []
        for idx in range(self.n_clients):
            msg = msgs[idx] if isinstance(msgs, list) else msgs
            t = Thread(target=utils.messaging.send_msg, args=(self.list_client_sockets[idx], msg))
            t.start()
            threads.append(t)
