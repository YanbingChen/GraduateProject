module IP4_Logger;

export {
    # Append the value LOG to the Log::ID enumerable.
    redef enum Log::ID += { LOG };

    # Define a new type called Factor::Info.
    type Info: record {
        uid:        	string &log;
        src_addr:		    addr &log;
        dest_addr:		    addr &log;
        header_length:		    count &log;
        type_of_service:	    count &log;
        total_length:           count &log;
        identification:         count &log;
        ttl:                    count &log;
        protocol:               count &log;
        };

    global log_package: function(c: connection, r: ip4_hdr);
    }
    
function log_package(c: connection, r: ip4_hdr)
    {
    Log::write(LOG, [$uid               =c$uid,
                    $src_addr           =r$src,
                    $dest_addr          =r$dst,
                    $header_length      =r$hl,
                    $type_of_service    =r$tos,
                    $total_length       =r$len,
                    $identification     =r$id,
                    $ttl                =r$ttl,
                    $protocol           =r$p]);
    }
