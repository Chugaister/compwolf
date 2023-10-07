import socket


def get_external_ip():
    hostname = socket.gethostname()
    external_ip = socket.gethostbyname(hostname)
    return external_ip
