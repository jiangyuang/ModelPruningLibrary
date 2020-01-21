from enum import Enum


class MessageTypes(Enum):
    ServerToClientInitMessage = 0
    ServerToClientUpdateMessage = 1
    ClientToServerAckMessage = 2
    ClientToServerUpdateMessage = 3


class BaseMessage:
    def __init__(self, msg_type: MessageTypes, data):
        self.msg_type = msg_type
        self.data = data

    def __getitem__(self, key):
        return self.data[key]

    def __str__(self):
        return str(self.msg_type)


class ServerToClientInitMessage(BaseMessage):
    def __init__(self, data: list):
        super(ServerToClientInitMessage, self).__init__(MessageTypes.ServerToClientInitMessage, data)

    @property
    def model(self):
        return self.data[0]

    @property
    def slice_indices(self):
        return self.data[1]

    @property
    def n_local_updates(self):
        return self.data[2]

    @property
    def batch_size(self):
        return self.data[3]

    @property
    def server_params(self):
        return self.data[4]


class ServerToClientUpdateMessage(BaseMessage):
    def __init__(self, data: list):
        super(ServerToClientUpdateMessage, self).__init__(MessageTypes.ServerToClientUpdateMessage, data)

    @property
    def state_dict(self):
        return self.data[0]

    @property
    def terminate(self) -> bool:
        return self.data[1]

    @property
    def use_dense(self) -> bool:
        return self.data[2]


class ClientToServerAckMessage(BaseMessage):
    def __init__(self):
        super(ClientToServerAckMessage, self).__init__(MessageTypes.ClientToServerAckMessage, None)

    def __getitem__(self, key):
        raise IndexError("ClientToServerAckMessage is not subscriptable")


class ClientToServerUpdateMessage(BaseMessage):
    def __init__(self, data: list):
        super(ClientToServerUpdateMessage, self).__init__(MessageTypes.ClientToServerUpdateMessage, data)

    @property
    def state_dict(self):
        return self.data[0]

    @property
    def data_size(self):
        return self.data[1]
