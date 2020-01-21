import socket
import struct
from utils.save_load import dumps, loads


def send_msg(sock, msg):
    msg_pickled = dumps(msg)
    sock.sendall(struct.pack(">I", len(msg_pickled)))
    sock.sendall(msg_pickled)
    # print("Sent ", msg, " to ", sock.getpeername())


def recv_msg(sock):
    msg_len = struct.unpack(">I", sock.recv(4))[0]
    msg = sock.recv(msg_len, socket.MSG_WAITALL)
    msg = loads(msg)
    # print("Received", msg, "from", sock.getpeername())
    return msg


def recv_msg_async(sock, ret: list, index):
    msg_len = struct.unpack(">I", sock.recv(4))[0]
    msg = sock.recv(msg_len, socket.MSG_WAITALL)
    msg = loads(msg)
    # print("Received", msg, "from", sock.getpeername())
    ret[index] = msg
