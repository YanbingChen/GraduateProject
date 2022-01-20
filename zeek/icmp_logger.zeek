module ICMP_Logger;

export {
    # Append the value LOG to the Log::ID enumerable.
    redef enum Log::ID += { LOG };

    # Define a new type called Factor::Info.
    type Info: record {
        uid:        	string &log;
        icmp_type:		count &log;
        };

    global log_package: function(c: connection, r: icmp_hdr);
    }
    
function log_package(c: connection, r: icmp_hdr)
    {
    Log::write(LOG, [$uid               =c$uid,
                    $icmp_type          =r$icmp_type]);
    }
