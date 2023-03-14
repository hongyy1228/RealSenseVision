import socket
import json


def client_program(arr):
    host = socket.gethostname()  # as both code is running on same pc
    port = 5000  # socket server port number

    client_socket = socket.socket()  # instantiate
    client_socket.connect((host, port))  # connect to the server


    data_string = json.dumps(arr)
    client_socket.send(data_string.encode())


    #data = client_socket.recv(4096)
    #data_arr = json.loads(data)
    # message = input(" -> ")  # take input
    #
    # while message.lower().strip() != 'bye':
    #     client_socket.send(message.encode())  # send message
    #     data = client_socket.recv(1024).decode()  # receive response
    #
    #     print('Received from server: ' + data)  # show in terminal
    #
    #     message = input(" -> ")  # again take input

    client_socket.close()  # close the connection
    #print(['Received', repr(data_arr)])


if __name__ == '__main__':
    client_program()