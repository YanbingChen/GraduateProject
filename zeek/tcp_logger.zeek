module TCP_Logger;

export {
    # Append the value LOG to the Log::ID enumerable.
    redef enum Log::ID += { LOG };

    # Define a new type called Factor::Info.
    type Info: record {
        uid:        	string &log;
        src_port:		port &log;
        dest_port:		port &log;
        seq:		    count &log;
        ack:	        count &log;
        header_length:  count &log;
        data_length:    count &log;
        reserved:       count &log;
        flags:          count &log;
        window:         count &log;
        };

    global log_package: function(c: connection, r: tcp_hdr);
    }
    
function log_package(c: connection, r: tcp_hdr)
    {
    Log::write(LOG, [$uid               =c$uid,
                    $src_port           =r$sport,
                    $dest_port          =r$dport,
                    $seq                =r$seq,
                    $ack                =r$ack,
                    $header_length      =r$hl,
                    $data_length        =r$dl,
                    $reserved           =r$reserved,
                    $flags              =r$flags,
                    $window             =r$win]);
    }
