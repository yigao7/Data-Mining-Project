# Implement the Bloom Filtering algorithm to estimate whether the user_id in the data stream has shown before.
# Find proper hash functions and the number of hash functions in the Bloom Filtering algorithm.
# Encapsulate the hash functions into a function called myhashs(s). The input of myhashs(s) is a string(user_id) and the output is a list of hash values.

import sys
from blackbox import BlackBox
from math import e, log
import binascii
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

def check_existence(stream_users):
    new = []
    for user in stream_users:
        hash_values = myhashs(user)
        for value in hash_values:
            if A[value] == 0:
                new.append(user)
                A[value] = 1
    return set(new)

input_filename = sys.argv[1]
stream_size = int(sys.argv[2])
num_of_asks = int(sys.argv[3])
output_filename = sys.argv[4]

n = 69997   # num of bits in array
m = stream_size * num_of_asks   # num of elements inserted
k = round(n / m * log(2))   # of hash functions

# generate hash functions
hash_function_list = []
for _ in range(k):
    a = random.randint(1, 100001)
    b = random.randint(0, 100000)
    hash_function_list.append(gen_hash_func(a,b,n))

if __name__ == '__main__':

    A = [0 for _ in range(n)]

    # FPR_est = (1 - e**(-k*m/n))**k
    # print(k, FPR_est)

    previous_users = set()

    bx = BlackBox()

    output = 'Time,FPR'
    for i in range(num_of_asks):
        stream_users = bx.ask(input_filename, stream_size)
        new = check_existence(stream_users)
        existed = set(stream_users) - new
        FPR = len(existed - previous_users.intersection(existed)) / stream_size
        previous_users = previous_users.union(stream_users)
        output += '\n'+str(i)+','+str(FPR)

    with open(output_filename, 'w+') as file:
        file.write(output)
    file.close()
