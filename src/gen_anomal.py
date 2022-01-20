import numpy as np
from timeline_mapping import TimelineMap
from log_reader import LogReader

import socket
from binascii import hexlify

import json

import random
random.seed(42)
np.random.seed(42)

def generate_anormal(normal_data, portion, field_interest, field_names):
    normal_shape = normal_data.shape
    anormal_count = int(normal_shape[0] * portion)
    anormal_shape = (anormal_count, normal_shape[1])

    rand_choise = np.random.choice(normal_data.shape[0], anormal_shape[0], replace=False)
    anormal_data = normal_data[rand_choise]

    for i in range(anormal_count):
        anormal_data[i, 0] = 0
        rand_num = random.randint(0, 32767)
        if rand_num & 1:
            # Change proto
            if anormal_data[i, field_names["proto"]] == 2.0:
                anormal_data[i, field_names["proto"]] -= random.randint(0, 2)
            elif anormal_data[i, field_names["proto"]] == 0.0:
                anormal_data[i, field_names["proto"]] += random.randint(0, 2)
            else:
                anormal_data[i, field_names["proto"]] += random.randint(-1, 1)
            pass
        rand_num >>= 1
        if rand_num & 1:
            # Change orig_pkts
            anormal_data[i, field_names["orig_pkts"]] += 1
            anormal_data[i, field_names["orig_pkts"]] *= random.randrange(0, 100) / 10.0
            pass
        rand_num >>= 1
        if rand_num & 1:
            # Change orig_ip_bytes
            anormal_data[i, field_names["orig_ip_bytes"]] += 1
            anormal_data[i, field_names["orig_ip_bytes"]] *= random.randrange(0, 100) / 10.0
            pass
        rand_num >>= 1
        if rand_num & 1:
            # Change resp_pkts
            anormal_data[i, field_names["resp_pkts"]] += 1
            anormal_data[i, field_names["resp_pkts"]] *= random.randrange(0, 100) / 10.0
            pass
        rand_num >>= 1
        if rand_num & 1:
            # Change resp_ip_bytes
            anormal_data[i, field_names["resp_ip_bytes"]] += 1
            anormal_data[i, field_names["resp_ip_bytes"]] *= random.randrange(0, 100) / 10.0
            pass
        rand_num >>= 1
        if rand_num & 1:
            # Change orig_ttl
            anormal_data[i, field_names["orig_ttl"]] += 1
            anormal_data[i, field_names["orig_ttl"]] *= random.randrange(0, 100) / 10.0
            pass
        rand_num >>= 1
        if rand_num & 1:
            # Change dest_ttl
            anormal_data[i, field_names["dest_ttl"]] += 1
            anormal_data[i, field_names["dest_ttl"]] *= random.randrange(0, 100) / 10.0
            pass
        rand_num >>= 1
        if rand_num & 1:
            # Change land
            anormal_data[i, field_names["land"]] = random.randint(0, 1)
            pass
        rand_num >>= 1
        if rand_num & 1:
            # Change orig_pkts_count
            anormal_data[i, field_names["orig_pkts"]] += 1
            anormal_data[i, field_names["orig_pkts"]] *= random.randrange(0, 100) / 10.0
            anormal_data[i, field_names["pkts_count"]] = anormal_data[i, field_names["orig_pkts"]] +\
                                                         anormal_data[i, field_names["orig_pkts"]]
            pass
        rand_num >>= 1
        if rand_num & 1:
            # Change dest_pkts_count
            anormal_data[i, field_names["resp_pkts"]] += 1
            anormal_data[i, field_names["resp_pkts"]] *= random.randrange(0, 100) / 10.0
            anormal_data[i, field_names["pkts_count"]] = anormal_data[i, field_names["resp_pkts"]] + \
                                                         anormal_data[i, field_names["resp_pkts"]]
            pass
        rand_num >>= 1
        if rand_num & 1:
            # Change period_length0
            anormal_data[i, field_names["period_length0"]] += 1
            anormal_data[i, field_names["period_length0"]] *= random.randrange(0, 100) / 10.0
            pass
        rand_num >>= 1
        if rand_num & 1:
            # Change period_length1
            anormal_data[i, field_names["period_length1"]] += 1
            anormal_data[i, field_names["period_length1"]] *= random.randrange(0, 100) / 10.0
            pass
        rand_num >>= 1
        if rand_num & 1:
            # Change period_length2
            anormal_data[i, field_names["period_length2"]] += 1
            anormal_data[i, field_names["period_length2"]] *= random.randrange(0, 100) / 10.0
            pass
        rand_num >>= 1
        if rand_num & 1:
            # Change period_length3
            anormal_data[i, field_names["period_length3"]] += 1
            anormal_data[i, field_names["period_length3"]] *= random.randrange(0, 100) / 10.0
            pass
        rand_num >>= 1
        if rand_num & 1:
            # Change period_length4
            anormal_data[i, field_names["period_length4"]] += 1
            anormal_data[i, field_names["period_length4"]] *= random.randrange(0, 100) / 10.0
            pass

        anormal_data[i, field_names["period_length0"]:(field_names["period_length4"]+1)].sort()

        ptr = field_names["period_length0"]
        if anormal_data[i, ptr] == 0.0:
            ptr2 = ptr + 1
            while ptr2 < anormal_shape[1] and anormal_data[i, ptr2] == 0.0:
                ptr2 += 1

            while ptr2 < anormal_shape[1]:
                anormal_data[i, ptr] = anormal_data[i, ptr2]
                ptr += 1
                ptr2 += 1

            while ptr < anormal_shape[1]:
                anormal_data[i, ptr] = 0.0
                ptr += 1
    return anormal_data
