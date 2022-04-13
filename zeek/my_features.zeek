module My_Features;

export 
{
    redef record Conn::Info += {
        orig_ttl:   count &default=0 &log;  #Log: eligable for log stream output
        dest_ttl:   count &default=0 &log;

        land:       bool &default=F &log;
        
        pkts_count:       count &default=0 &log;

        min_IAT:    double &default=0.0 &log;
        max_IAT:    double &default=0.0 &log;
        mean_IAT:   double &default=0.0 &log;
    };
    
    # Define a new type called Factor::Info.
    type Features: record {
        orig_ttl:       count;
        dest_ttl:		count;

        orig_pkts:      count;
        dest_pkts:      count;

        min_IAT:        double;
        max_IAT:        double;
        mean_IAT:       double;
        };

    # TTL part
    type TTL_Info: record {
        src_add: addr;
        src_times: count &default=0;
        src_ttl_sum: count &default=0;
        dest_add: addr;
        dest_times: count &default=0;
        dest_ttl_sum: count &default=0;
    };

    # TTL Average result
    type TTL_Average: record {
        orig_ttl: count;
        dest_ttl: count;
    };

    # A table, indexed by string, to TTL_Info
    type TTLTable: table[string] of TTL_Info;

    # Global declearation
    global ttl_table: TTLTable;

    global record_ttl: function(c: connection, p: pkt_hdr);
    global extract_ttl: function(c: connection): TTL_Average;


    # -------- Package part --------

    type Package_Info: record {
        src_addr: addr;
        dest_addr: addr;
        timestamp: time;
        
    };

    type Package_Summary: record {
        orig_pkts: count;
        resp_pkts: count;
        min_IAT: double;
        max_IAT: double;
        mean_IAT: double;
    };

    type Package_Vec: vector of Package_Info;
    type PackageTable: table[string] of Package_Vec;

    global package_table: PackageTable;

    global extract_info: function(c: connection): Package_Summary;

    global global_count: count;

    # -------- Summary --------
    global record_package: function(c: connection, p: pkt_hdr);
    global extract_feature: function(c: connection): Features;
    global summaryConnection: function(c: connection);
}

# Called before a connection closed and outputed
function summaryConnection(c: connection)
{
    # TTL average
    local f: Features = extract_feature(c);     # CALL: extract_feature
    c$conn$orig_ttl = f$orig_ttl;
    c$conn$dest_ttl = f$dest_ttl;

    # Land
    if(c$id$orig_h == c$id$resp_h) {
        c$conn$land = T;
    }

    # Pkts count
    c$conn$pkts_count = f$dest_pkts + f$orig_pkts;

    # IATs
    c$conn$min_IAT = f$min_IAT;
    c$conn$max_IAT = f$max_IAT;
    c$conn$mean_IAT = f$mean_IAT;
}

# Extract desired features of a connection
function extract_feature(c: connection): Features
{
    # TTL
    local average: TTL_Average = extract_ttl(c);    # CALL: extract_ttl

    # Package
    local pkg_info: Package_Summary = extract_info(c);  # CALL: extract_info
    
    return Features($orig_ttl = average$orig_ttl, $dest_ttl = average$dest_ttl, 
                    $orig_pkts = pkg_info$orig_pkts, $dest_pkts = pkg_info$resp_pkts,
                    $min_IAT = pkg_info$min_IAT, $max_IAT = pkg_info$max_IAT, $mean_IAT = pkg_info$mean_IAT);
}


function extract_ttl(c: connection): TTL_Average {
    local ttl1: count = 0;
    local ttl2: count = 0;

    # Calculate average TTL
    if(c$uid in ttl_table) {
        if(ttl_table[c$uid]$src_times != 0) {
            ttl1 = ttl_table[c$uid]$src_ttl_sum / ttl_table[c$uid]$src_times;
        }
        
        if(ttl_table[c$uid]$dest_times != 0) {
            ttl2 = ttl_table[c$uid]$dest_ttl_sum / ttl_table[c$uid]$dest_times;
        }
        
    }
    return TTL_Average($orig_ttl = ttl1, $dest_ttl = ttl2);
}

function extract_info(c: connection): Package_Summary
{
    local pkg_cnt1: count = 0;
    local pkg_cnt2: count = 0;
    
    local init_flag: bool = F;
    local min_IAT: double = -1.0;
    local max_IAT: double = 0.0;
    local mean_IAT: double = 0.0;
    local start_time: time;
    local last_time: time;

    # Count orig_pkts & resp_pkts
    for(p in package_table[c$uid]) {
        if(init_flag==F) {
            init_flag = T;
            start_time = package_table[c$uid][p]$timestamp;
        } else {
            local gap : double = interval_to_double(package_table[c$uid][p]$timestamp - last_time);
            if(min_IAT > gap || min_IAT < 0.0) {
                min_IAT = gap;
            }
            if(max_IAT < gap) {
                max_IAT = gap;
            }
        }

        last_time = package_table[c$uid][p]$timestamp;

        if(package_table[c$uid][p]$src_addr == c$id$orig_h) {
            ++pkg_cnt1;
        } else {
            ++pkg_cnt2;
        }
    }
    
    mean_IAT = interval_to_double(last_time - start_time) / (pkg_cnt1 + pkg_cnt2);

    if(min_IAT < 0.0) {
        min_IAT = 0.0;
    }

    return Package_Summary($orig_pkts=pkg_cnt1, $resp_pkts=pkg_cnt2, $min_IAT=min_IAT, $max_IAT=max_IAT, $mean_IAT=mean_IAT);
}

# For each package, record its features
function record_package(c: connection, p: pkt_hdr)
{
    
    if(p ?$ ip) {
        # TTL
        record_ttl(c, p);
        
        # Package
        if(c$uid !in package_table) {
            package_table[c$uid] = vector(Package_Info($src_addr=p$ip$src, $dest_addr=p$ip$dst, $timestamp=network_time()));
        } else {
            package_table[c$uid] += Package_Info($src_addr=p$ip$src, $dest_addr=p$ip$dst, $timestamp=network_time());
        }
    }

    

    ++global_count;
    
    if(global_count % 100000 == 0) {
        print fmt("Processed: %llu", global_count);
    }

}

function record_ttl(c: connection, p: pkt_hdr) 
{
    if(c$uid !in ttl_table) {
        ttl_table[c$uid] = [$src_add = c$id$orig_h, $dest_add = c$id$resp_h];
    }
    
    if(p$ip$src == c$id$orig_h) {
        ++(ttl_table[c$uid]$src_times);
        ttl_table[c$uid]$src_ttl_sum += p$ip$ttl;
    } else {
        ++(ttl_table[c$uid]$dest_times);
        ttl_table[c$uid]$dest_ttl_sum += p$ip$ttl;
    }
    
}
