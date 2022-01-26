# Implement the fixed size sampling method (Reservoir Sampling Algorithm).
# Assume that the memory can only save 100 users, so use the fixed size sampling method to only keep part of the users as a sample in the streaming. 
# When the streaming of the users comes, for the first 100 users, directly save them in a list. 
# After that, for the nth (starting from 1) user in the whole sequence of users, keep the nth user with the probability of 100/n, otherwise discard it. 
# If keep the nth user, randomly pick one in the list to be replaced.

import sys
from blackbox import BlackBox
import random

if __name__ == '__main__':

    input_filename = sys.argv[1]
    stream_size = int(sys.argv[2])
    num_of_asks = int(sys.argv[3])
    output_filename = sys.argv[4]

    random.seed(553)

    s = 100  # Reservoir size

    bx = BlackBox()

    output = 'seqnum,0_id,20_id,40_id,60_id,80_id'

    n = 0
    S = []
    for i in range(num_of_asks):
        output += '\n' + str((i+1)*stream_size) + ','
        stream_users = bx.ask(input_filename, s)
        for user in stream_users:
            n += 1
            if n <= s:
                S.append(user)
                continue
            keep = random.random()
            if keep < s/n:
                index = random.randint(0, s-1)
                S[index] = user
        output += ','.join([S[0], S[20], S[40], S[60], S[80]])

        with open(output_filename, 'w+') as file:
            file.write(output)
        file.close()
