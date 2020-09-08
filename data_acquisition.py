#------------------------------------------------------------------------------------------------------------------
#   Sample program for data acquisition and recording.
#------------------------------------------------------------------------------------------------------------------
import time
import socket
import numpy as np

# Data configuration
n_channels = 5
samp_rate = 256
emg_data = [[] for i in range(n_channels)]
samp_count = 0

# Socket configuration
UDP_IP = '127.0.0.1'
UDP_PORT = 8000
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.settimeout(0.01)

# Data acquisition
start_time = time.time()

while True:
    try:
        data, addr = sock.recvfrom(1024*1024)                        
            
        values = np.frombuffer(data)       
        ns = int(len(values)/n_channels)
        samp_count+=ns        

        for i in range(ns):
            for j in range(n_channels):
                emg_data[j].append(values[n_channels*i + j])
            
        elapsed_time = time.time() - start_time
        if (elapsed_time > 0.1):
            start_time = time.time()
            print ("Muestras: ", ns)
            print ("Cuenta: ", samp_count)
            print ("Última lectura: ", [row[samp_count-1] for row in emg_data])
            print("")
            
    except socket.timeout:
        pass  
    
    

