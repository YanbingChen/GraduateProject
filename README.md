# Vanilla_GradProject
A Graduation project repository for PKU

# Directories

- **/out/zeek_logs**: All zeek logs from different .pcap packs stores here
  - **Summary.log**: Concatenated logs.

- **/out/dataset**: Datasets generated using zeek_logs. Subdirectories are for different algorithms.

- **/out/period_output**: Generated period data using period detection data.
  - **robust_period.csv**: Period data from robust_period method.
  - **SummaryWithPeriod.log**: Summary.log with all period data.

- **/out/models**: Stores all trained models.

- **/zeek/MyLog.sh**: A bash script automatically runs zeek scripts on given pcap files.

# Running sequence

### Pre processing

- Run **zeek** on each pcap package to generate conn.log. 
  - /zeek % `./MyLog.sh pcap/smvs.pcap1533 pcap/smvs.pcap1534 pcap/smvs.pcap1535 pcap/smvs.pcap1536 pcap/smvs.pcap1537 pcap/smvs.pcap1538 pcap/smvs.pcap1539`
- Concatenate different logs to one single log.
  - /src % `python concatenate_logs.py ../out/zeek_logs ../out/zeek_logs/Summary.log`
  - TODO commandline param
- Run **robust_period** algorithm. It uses logs from **/out/zeek_logs** directory.
  - /src % `python gen_period_data.py ../out/zeek_logs/Summary.log ../out/period_output/SummaryWithPeriod.log ../out/period_output`
- Prepare datasets for different nn to run.
  - /src % `python dataset.py ../out/period_output/SummaryWithPeriod.log ../out/dataset`

### Training
- Train NN.
  - /src % `python nn_train.py ../out/dataset ../out/models`
- Train DROCC.
  - /src % `python drocc_train.py --data_path ../out/dataset --model_dir ../out/models  --only_ce_epochs 10 --metric F1`

# Test Notes

#### Zeek running time: 222 seconds

#### Robust period time: 1849 seconds on 13256 connection records (394 events, in which 304 used for periodicity)

####
