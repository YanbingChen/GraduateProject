module UDP_Logger;

export {
    # Append the value LOG to the Log::ID enumerable.
    redef enum Log::ID += { LOG };

    # Define a new type called Factor::Info.
    type Info: record {
        uid:        	string &log;
        src_port:		port &log;
        dest_port:		port &log;
        udp_length:		count &log;
        };

    global log_package: function(c: connection, r: udp_hdr);
    }
    
function log_package(c: connection, r: udp_hdr)
    {
    Log::write(LOG, [$uid               =c$uid,
                    $src_port           =r$sport,
                    $dest_port          =r$dport,
                    $udp_length         =r$ulen]);
    }
