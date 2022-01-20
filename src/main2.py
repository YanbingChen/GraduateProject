import numpy as np
from timeline_mapping import TimelineMap
from log_reader import LogReader
from gen_anomal import generate_anormal

import socket
from binascii import hexlify

import json

def IPV6_to_int(ipv6_addr):
    return int(hexlify(socket.inet_pton(socket.AF_INET6, ipv6_addr)), 16)

# ------ Pre process & Timeline mapping ------
logreader = LogReader("../zeek/conn.log", ["id.orig_h", "id.resp_h"])
tm = TimelineMap()
for i in logreader:
    tm.insert(i.unique_id(), i.get_timestamp())

# Load period flags and data from file
event_period_flag = {}
event_period_data = {}
with open("../out/event_period_flag_temp.txt", 'r') as f:
    event_period_flag = json.load(f)
with open("../out/event_period_data_temp.txt", 'r') as f:
    event_period_data = json.load(f)

# Add calculated fields to LogRecord
logreader.add_field("period")
logreader.add_field("period_length0")
logreader.add_field("period_length1")
logreader.add_field("period_length2")
logreader.add_field("period_length3")
logreader.add_field("period_length4")
for record in logreader:
    uid = record.uid()
    flag = event_period_flag[uid]
    record.add_value(str(flag))
    lengths = event_period_data[uid]
    for i in range(5):
        if i < len(lengths):
            record.add_value(str(lengths[i]))
        else:
            record.add_value("0.0")

# ----------------  Convert to ML-ready format  --------------------
# Define interest fields
field_interest = ['id.orig_h', 'id.resp_h', 'proto', 'orig_pkts',
                  'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes',
                  'orig_ttl', 'dest_ttl', 'land', 'duration',
                  'orig_pkts', 'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes', 'pkts_count'
                  , 'period', 'period_length0', 'period_length1',
                  'period_length2', 'period_length3', 'period_length4'
                  ]

field_names = logreader.get_fields()

id_record_index = {}
factory_id = 1
# Create numpy array for storage
result = np.zeros([len(logreader), len(field_interest) + 2])
for rec_num, record in enumerate(logreader):
    result[rec_num, 0] = 1
    result[rec_num, 1] = factory_id
    id_record_index[factory_id] = record[1]
    factory_id += 1

    for i, val in enumerate(field_interest):
        field_id = field_names[val]
        fieldContent = record[field_id]
        if val == "id.orig_h" or val == "id.resp_h":
            splits = fieldContent.split(".")
            if len(splits) > 1:
                fieldContent = 0
                for s in splits:
                    fieldContent *= 256
                    fieldContent += int(s)
            else:
                fieldContent = 3232235777
        elif val == "proto":
            if fieldContent == "tcp":
                fieldContent = 0
            elif fieldContent == "udp":
                fieldContent = 1
            else:
                fieldContent = 2
        elif val == "land":
            if fieldContent == "F":
                fieldContent = 0
            else:
                fieldContent = 1
        else:
            if fieldContent == '-':
                fieldContent = "0.0"
            fieldContent = float(fieldContent)

        result[rec_num, i+2] = fieldContent

# Store numpy
np.save("../out/cleanData.npy", result)

new_namemap = {}
for i in range(len(field_interest)):
    new_namemap[field_interest[i]] = i + 2

anormal_result = generate_anormal(result, 0.111, field_interest, new_namemap)

combine_result = np.concatenate((result, anormal_result))

# Normalize dataset
featureAve = np.average(combine_result, axis=0)
featureStd = np.std(combine_result, axis=0)
combine_result[:, 2:] -= featureAve[2:]
combine_result[:, 2:] /= featureStd[2:]

# Save npy
np.save("../out/fullDataset.npy", combine_result)
np.save("../out/featureAve.npy", featureAve)
np.save("../out/featureStd.npy", featureStd)

# Save event map
with open("../out/id_feature.txt", 'w') as f:
    json.dump(id_record_index, f)

