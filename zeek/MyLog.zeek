@load ip4_logger
@load tcp_logger
@load udp_logger
@load icmp_logger
@load my_features

# Before processing packages
event zeek_init()
{
    # Create the logging stream.
    Log::create_stream(IP4_Logger::LOG, [$columns=IP4_Logger::Info, $path="ip4_packs"]);
    Log::create_stream(TCP_Logger::LOG, [$columns=TCP_Logger::Info, $path="tcp_packs"]);
    Log::create_stream(UDP_Logger::LOG, [$columns=UDP_Logger::Info, $path="udp_packs"]);
    Log::create_stream(ICMP_Logger::LOG, [$columns=ICMP_Logger::Info, $path="icmp_packs"]);

    My_Features::global_count = 0;
}

# After packages processed
event zeek_done()
{
    
}

# Connection summary: when connection closed
event connection_state_remove(c: connection)
{
    My_Features::summaryConnection(c);
}

# Called when a new package transfered. (called for each package)
event new_packet(c: connection, p: pkt_hdr) 
{
    if (p ?$ ip) {
        IP4_Logger::log_package(c, p$ip);
    }

    if (p ?$ tcp) {
        TCP_Logger::log_package(c, p$tcp);
    }

    if (p ?$ udp) {
        UDP_Logger::log_package(c, p$udp);
    }

    if (p ?$ icmp) {
        ICMP_Logger::log_package(c, p$icmp);
    }
    
    # Statistic features extraction
    My_Features::record_package(c, p);
}