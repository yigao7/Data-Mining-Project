# Implement the Flajolet-Martin algorithm to estimate the number of unique users within a window in the data stream.
  
import sys
from blackbox import BlackBox
import binascii
from statistics import mean, median
import random

def gen_hash_func(a, b, m):
    def hash_func(x):
        return (a * int(binascii.hexlify(x.encode('utf8')),16) + b) % m
    return hash_func

def myhashs(s):
    result = []
    for f in hash_function_list:
        result.append(f(s))
    return result

def flajolet_martin(stream_users):
    Rs = []

    for user in stream_users:
        R = []
        hash_values = myhashs(user)
        for value in hash_values:
            bit = '{0:010b}'.format(value)
            R.append(len(bit) - len(bit.rstrip('0')))
        Rs.append(R)

    max_Rs = [max([i[j] for i in Rs]) for j in range(len(Rs[0]))]
    max_R_avg = []
    for i in range(group_hash):
        group = max_Rs[i * int(num_of_hash/group_hash):(i + 1) * int(num_of_hash/group_hash)]
        max_R_avg.append(mean(group))
    return round(2**(median(max_R_avg)))

input_filename = sys.argv[1]
stream_size = int(sys.argv[2])
num_of_asks = int(sys.argv[3])
output_filename = sys.argv[4]

m = 69997
num_of_hash = 200
group_hash = 5

# generate hash functions
hash_function_list = []
for _ in range(num_of_hash):
    a = random.randint(1, 100001)
    b = random.randint(0, 100000)
    hash_function_list.append(gen_hash_func(a, b, m))

if __name__ == '__main__':

    bx = BlackBox()

    output = 'Time,Ground Truth,Estimation'

    for i in range(num_of_asks):
        stream_users = bx.ask(input_filename, stream_size)
        estimate = flajolet_martin(stream_users)
        output += '\n' + str(i) + ',' + str(len(set(stream_users))) + ',' + str(estimate)

    with open(output_filename, 'w+') as file:
        file.write(output)
    file.close()
