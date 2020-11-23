#------------------------------------------------------------------------------------------------------------------
#   Program that simulates the acquisition and transmision of EMG data.
#------------------------------------------------------------------------------------------------------------------
import numpy as np
import socket
import time
import struct

# EMG data
samp_rate = 256

data = np.loadtxt("Izq_Der_Cerr_Abi_1.txt") 
mark = data[:, data.shape[1]-1]
samps = data.shape[0]

data = np.delete(data, data.shape[1]-1, 1)
data = np.delete(data, 5, 1)
data = np.delete(data, 4, 1)
data = np.delete(data, 2, 1)
data = np.delete(data, 0, 1)
n_channels = data.shape[1]
# Socket configuration
UDP_IP = '127.0.0.1'
UDP_PORT = 8080

# Send data
start_index = 0
end_index = 0
start_time = time.time()
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((UDP_IP, UDP_PORT))
    print('Waiting for client...')
    s.listen(5)                 # Now wait for client connection.
    conn, addr = s.accept()     # Establish connection with client.
    with conn:
        print('Recieved connection from:', addr)
        while True:
            # Calculate elapsed time and current data index
            time.sleep(0.05)
            elapsed_time = time.time() - start_time

            start_index = end_index
            end_index = int(samp_rate*elapsed_time)

            index = start_index % samps
            ns = end_index - start_index

            # Send data for the calculated range
            if (ns > 0):

                # Build data package
                out_data = []
                for i in range(ns):
                    for j in range(n_channels):
                        out_data.append(data[index][j])            

                    if (mark[index] != 0):
                        print('--------------- Marca:', int(mark[index]), '---------------')

                    index+=1
                    index %= samps
                pack_string = '<' + str(n_channels*ns) + 'd'
                bin_data = struct.pack(pack_string, *out_data)

                # Send data
                conn.send(bin_data)

        #print("Muestras enviadas:", ns)