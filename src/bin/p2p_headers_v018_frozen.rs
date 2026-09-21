use bitcoin::{
    block::Header,
    consensus::{
        deserialize,
        encode::serialize,
    },
    hashes::{sha256d, Hash},
    p2p::{
        address::Address,
        message::{NetworkMessage, RawNetworkMessage},
        message_blockdata::GetHeadersMessage,
        message_network::VersionMessage,
        Magic,
        ServiceFlags,
    },
    BlockHash,
};

use csv::Reader;
use rand::RngExt;

use std::{
    error::Error,
    io::{Cursor, Read, Write},
    net::{SocketAddr, TcpStream, ToSocketAddrs},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

const INPUT: &str = "data/bitcoin-blocks.csv";
const TARGET_HEIGHT: u64 = 899_999;
const MAX_HEADERS_PER_BATCH: usize = 2000;
const MAX_P2P_PAYLOAD: usize = 32 * 1024 * 1024;


// ============================================================
// RECEIVED P2P FRAME
// ============================================================

struct P2pFrame {
    command: String,
    payload: Vec<u8>,
}


// ============================================================
// TIME
// ============================================================

fn unix_time() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64
}


// ============================================================
// DATASET ANCHOR
// ============================================================

fn last_verified_anchor(
) -> Result<(u64, BlockHash), Box<dyn Error>> {
    let mut reader = Reader::from_path(INPUT)?;

    let columns = reader.headers()?.clone();

    let height_idx = columns
        .iter()
        .position(|x| x == "height")
        .ok_or("missing height column")?;

    let hash_idx = columns
        .iter()
        .position(|x| x == "block_hash")
        .ok_or("missing block_hash column")?;

    let verified_idx = columns
        .iter()
        .position(|x| x == "verified")
        .ok_or("missing verified column")?;

    let mut best: Option<(u64, BlockHash)> = None;

    for row in reader.records() {
        let row = row?;

        if row.get(verified_idx) != Some("true") {
            continue;
        }

        let height: u64 = row
            .get(height_idx)
            .ok_or("missing height")?
            .parse()?;

        let hash: BlockHash = row
            .get(hash_idx)
            .ok_or("missing block hash")?
            .parse()?;

        match best {
            None => best = Some((height, hash)),

            Some((old_height, _))
                if height > old_height =>
            {
                best = Some((height, hash));
            }

            _ => {}
        }
    }

    best.ok_or_else(|| "no verified anchor found".into())
}


// ============================================================
// PEER DISCOVERY
// ============================================================

fn resolve_peer(
) -> Result<SocketAddr, Box<dyn Error>> {
    let addresses =
        ("seed.bitcoin.sipa.be", 8333).to_socket_addrs()?;

    let peers: Vec<SocketAddr> = addresses.collect();

    if let Some(peer) = peers.iter().find(|p| p.is_ipv4()) {
        return Ok(*peer);
    }

    peers
        .first()
        .copied()
        .ok_or_else(|| "DNS seed returned no peers".into())
}


// ============================================================
// SEND P2P MESSAGE
// ============================================================

fn send_message(
    stream: &mut TcpStream,
    payload: NetworkMessage,
) -> Result<(), Box<dyn Error>> {
    let message =
        RawNetworkMessage::new(
            Magic::BITCOIN,
            payload,
        );

    let bytes = serialize(&message);

    stream.write_all(&bytes)?;
    stream.flush()?;

    Ok(())
}


// ============================================================
// READ RAW BITCOIN P2P FRAME
// ============================================================

fn read_frame(
    stream: &mut TcpStream,
) -> Result<P2pFrame, Box<dyn Error>> {
    let mut envelope = [0u8; 24];

    stream.read_exact(&mut envelope)?;

    // --------------------------------------------------------
    // Mainnet magic
    // --------------------------------------------------------

    let expected_magic =
        Magic::BITCOIN.to_bytes();

    if envelope[0..4] != expected_magic {
        return Err(
            format!(
                "invalid Bitcoin magic: {:02x?}",
                &envelope[0..4]
            )
            .into(),
        );
    }

    // --------------------------------------------------------
    // Command
    // --------------------------------------------------------

    let command_bytes = &envelope[4..16];

    let command_end =
        command_bytes
            .iter()
            .position(|b| *b == 0)
            .unwrap_or(command_bytes.len());

    let command =
        std::str::from_utf8(
            &command_bytes[..command_end],
        )?
        .to_string();

    // --------------------------------------------------------
    // Payload length
    // --------------------------------------------------------

    let payload_len =
        u32::from_le_bytes([
            envelope[16],
            envelope[17],
            envelope[18],
            envelope[19],
        ]) as usize;

    if payload_len > MAX_P2P_PAYLOAD {
        return Err(
            format!(
                "unreasonable P2P payload: {} bytes",
                payload_len
            )
            .into(),
        );
    }

    // --------------------------------------------------------
    // Payload
    // --------------------------------------------------------

    let mut payload =
        vec![0u8; payload_len];

    if payload_len > 0 {
        stream.read_exact(&mut payload)?;
    }

    // --------------------------------------------------------
    // Bitcoin message checksum = first 4 bytes SHA256d(payload)
    // --------------------------------------------------------

    let digest =
        sha256d::Hash::hash(&payload);

    let digest_bytes =
        digest.to_byte_array();

    if digest_bytes[0..4] != envelope[20..24] {
        return Err(
            format!(
                "checksum mismatch for '{}'",
                command
            )
            .into(),
        );
    }

    Ok(P2pFrame {
        command,
        payload,
    })
}


// ============================================================
// COMPACTSIZE
// ============================================================

fn read_compact_size<R: Read>(
    reader: &mut R,
) -> Result<u64, Box<dyn Error>> {
    let mut first = [0u8; 1];

    reader.read_exact(&mut first)?;

    match first[0] {
        0x00..=0xfc => {
            Ok(first[0] as u64)
        }

        0xfd => {
            let mut buf = [0u8; 2];
            reader.read_exact(&mut buf)?;

            Ok(
                u16::from_le_bytes(buf)
                    as u64
            )
        }

        0xfe => {
            let mut buf = [0u8; 4];
            reader.read_exact(&mut buf)?;

            Ok(
                u32::from_le_bytes(buf)
                    as u64
            )
        }

        0xff => {
            let mut buf = [0u8; 8];
            reader.read_exact(&mut buf)?;

            Ok(
                u64::from_le_bytes(buf)
            )
        }
    }
}


// ============================================================
// HEADERS PAYLOAD DECODER
// ============================================================
//
// Bitcoin "headers" payload:
//
// CompactSize count
//
// repeated count times:
//
//     80-byte block header
//     CompactSize transaction_count
//
// For "headers", transaction_count MUST be zero.
//
// Example for 2000 headers:
//
//     count = fd d0 07          3 bytes
//     2000 * (80 + 1)     162000 bytes
//                         ------------
//                         162003 bytes
//
// Exactly what our peer sent.
//
// ============================================================

fn decode_headers_payload(
    payload: &[u8],
) -> Result<Vec<Header>, Box<dyn Error>> {
    let mut cursor =
        Cursor::new(payload);

    let count =
        read_compact_size(&mut cursor)?;

    if count > MAX_HEADERS_PER_BATCH as u64 {
        return Err(
            format!(
                "peer sent {} headers; protocol maximum expected {}",
                count,
                MAX_HEADERS_PER_BATCH
            )
            .into(),
        );
    }

    let mut headers =
        Vec::with_capacity(count as usize);

    for index in 0..count {
        // ----------------------------------------------------
        // Exact Bitcoin block header: 80 bytes
        // ----------------------------------------------------

        let mut header_bytes =
            [0u8; 80];

        cursor.read_exact(
            &mut header_bytes,
        )?;

        let header: Header =
            deserialize(&header_bytes)?;

        // ----------------------------------------------------
        // Transaction count follows each header.
        //
        // In a HEADERS message this must be zero.
        // ----------------------------------------------------

        let tx_count =
            read_compact_size(&mut cursor)?;

        if tx_count != 0 {
            return Err(
                format!(
                    "header {} has non-zero tx_count {}",
                    index,
                    tx_count
                )
                .into(),
            );
        }

        headers.push(header);
    }

    // --------------------------------------------------------
    // Entire payload must have been consumed.
    // --------------------------------------------------------

    let consumed =
        cursor.position() as usize;

    if consumed != payload.len() {
        return Err(
            format!(
                "headers parser consumed {} of {} bytes; {} bytes remain",
                consumed,
                payload.len(),
                payload.len() - consumed
            )
            .into(),
        );
    }

    Ok(headers)
}


// ============================================================
// HANDSHAKE
// ============================================================

fn handshake(
    stream: &mut TcpStream,
    peer: SocketAddr,
) -> Result<(), Box<dyn Error>> {
    let services =
        ServiceFlags::NONE;

    let receiver =
        Address::new(
            &peer,
            services,
        );

    let sender_addr: SocketAddr =
        "0.0.0.0:0".parse()?;

    let sender =
        Address::new(
            &sender_addr,
            services,
        );

    let nonce: u64 =
        rand::rng().random();

    let version =
        VersionMessage::new(
            services,
            unix_time(),
            receiver,
            sender,
            nonce,
            "/X1331:0.17/".to_string(),
            0,
        );

    send_message(
        stream,
        NetworkMessage::Version(version),
    )?;

    let mut got_version = false;
    let mut got_verack = false;

    while !(got_version && got_verack) {
        let frame =
            read_frame(stream)?;

        match frame.command.as_str() {
            "version" => {
                let v: VersionMessage =
                    deserialize(&frame.payload)?;

                println!(
                    "Peer version: {} | user-agent: {} | height: {}",
                    v.version,
                    v.user_agent,
                    v.start_height
                );

                got_version = true;

                send_message(
                    stream,
                    NetworkMessage::Verack,
                )?;
            }

            "verack" => {
                println!(
                    "Peer verack received."
                );

                got_verack = true;
            }

            "ping" => {
                let ping_nonce: u64 =
                    deserialize(&frame.payload)?;

                send_message(
                    stream,
                    NetworkMessage::Pong(
                        ping_nonce,
                    ),
                )?;
            }

            other => {
                println!(
                    "Handshake ignored message: {}",
                    other
                );
            }
        }
    }

    Ok(())
}


// ============================================================
// REQUEST HEADERS
// ============================================================

fn request_headers(
    stream: &mut TcpStream,
    locator: BlockHash,
) -> Result<Vec<Header>, Box<dyn Error>> {
    let request =
        GetHeadersMessage::new(
            vec![locator],
            BlockHash::all_zeros(),
        );

    send_message(
        stream,
        NetworkMessage::GetHeaders(request),
    )?;

    loop {
        let frame =
            read_frame(stream)?;

        match frame.command.as_str() {
            "headers" => {
                println!(
                    "Received 'headers': {} bytes",
                    frame.payload.len()
                );

                let headers =
                    decode_headers_payload(
                        &frame.payload,
                    )?;

                println!(
                    "Decoded {} Bitcoin headers.",
                    headers.len()
                );

                return Ok(headers);
            }

            "ping" => {
                let ping_nonce: u64 =
                    deserialize(&frame.payload)?;

                send_message(
                    stream,
                    NetworkMessage::Pong(
                        ping_nonce,
                    ),
                )?;
            }

            other => {
                println!(
                    "Ignored P2P message: {} ({} bytes)",
                    other,
                    frame.payload.len()
                );
            }
        }
    }
}


// ============================================================
// VERIFY HEADER BATCH
// ============================================================

fn verify_batch(
    previous: BlockHash,
    headers: &[Header],
) -> Result<(), Box<dyn Error>> {
    let mut expected_previous =
        previous;

    for (index, header)
        in headers.iter().enumerate()
    {
        // ----------------------------------------------------
        // Chain continuity
        // ----------------------------------------------------

        if header.prev_blockhash
            != expected_previous
        {
            return Err(
                format!(
                    "broken chain at header {}: expected prev {}, got {}",
                    index,
                    expected_previous,
                    header.prev_blockhash
                )
                .into(),
            );
        }

        // ----------------------------------------------------
        // Local Proof-of-Work verification
        // ----------------------------------------------------

        header
            .validate_pow(
                header.target(),
            )
            .map_err(|e| {
                format!(
                    "invalid PoW at header {}: {}",
                    index,
                    e
                )
            })?;

        expected_previous =
            header.block_hash();
    }

    Ok(())
}


// ============================================================
// MAIN
// ============================================================

fn main(
) -> Result<(), Box<dyn Error>> {
    println!(
        "X1331 Bitcoin P2P Header Collector"
    );

    println!(
        "=================================="
    );

    let (
        anchor_height,
        anchor_hash,
    ) = last_verified_anchor()?;

    println!(
        "Dataset       : {}",
        INPUT
    );

    println!(
        "Anchor height : {}",
        anchor_height
    );

    println!(
        "Anchor hash   : {}",
        anchor_hash
    );

    println!(
        "Target height : {}",
        TARGET_HEIGHT
    );

    if anchor_height >= TARGET_HEIGHT {
        println!(
            "Target already reached."
        );

        return Ok(());
    }

    // --------------------------------------------------------
    // Peer discovery
    // --------------------------------------------------------

    let peer =
        resolve_peer()?;

    println!(
        "Peer          : {}",
        peer
    );

    println!(
        "Connecting..."
    );

    let mut stream =
        TcpStream::connect_timeout(
            &peer,
            Duration::from_secs(10),
        )?;

    stream.set_read_timeout(
        Some(Duration::from_secs(30)),
    )?;

    stream.set_write_timeout(
        Some(Duration::from_secs(30)),
    )?;

    // --------------------------------------------------------
    // P2P handshake
    // --------------------------------------------------------

    handshake(
        &mut stream,
        peer,
    )?;

    println!();

    println!(
        "Handshake complete."
    );

    println!(
        "Requesting headers after {}...",
        anchor_height
    );

    // --------------------------------------------------------
    // Synchronization
    // --------------------------------------------------------

    let mut locator =
        anchor_hash;

    let mut height =
        anchor_height;

    let mut total =
        0usize;

    let expected_total =
        TARGET_HEIGHT - anchor_height;

    while height < TARGET_HEIGHT {
        let headers =
            request_headers(
                &mut stream,
                locator,
            )?;

        if headers.is_empty() {
            println!(
                "Peer returned no more headers."
            );

            break;
        }

        // ----------------------------------------------------
        // Verify linkage + PoW
        // ----------------------------------------------------

        verify_batch(
            locator,
            &headers,
        )?;

        let remaining =
            TARGET_HEIGHT - height;

        let usable =
            std::cmp::min(
                headers.len(),
                remaining as usize,
            );

        let batch_start =
            height + 1;

        let batch_end =
            height + usable as u64;

        println!(
            "Verified {:4} headers: {} .. {}",
            usable,
            batch_start,
            batch_end
        );

        // ----------------------------------------------------
        // Consume verified headers
        // ----------------------------------------------------

        for header
            in headers.iter().take(usable)
        {
            height += 1;
            total += 1;

            let hash =
                header.block_hash();

            // Print first five as sanity check.
            if height
                <= anchor_height + 5
            {
                println!(
                    "  {} | nonce {:10} | {}",
                    height,
                    header.nonce,
                    hash
                );
            }

            locator =
                hash;

            if height >= TARGET_HEIGHT {
                break;
            }
        }

        if height >= TARGET_HEIGHT {
            break;
        }

        if headers.len()
            < MAX_HEADERS_PER_BATCH
        {
            println!(
                "Peer returned only {} headers.",
                headers.len()
            );

            println!(
                "Likely reached peer chain tip."
            );

            break;
        }
    }

    // --------------------------------------------------------
    // Final report
    // --------------------------------------------------------

    println!();

    println!(
        "P2P COLLECTION TEST COMPLETE"
    );

    println!(
        "============================"
    );

    println!(
        "Anchor           : {}",
        anchor_height
    );

    println!(
        "Expected headers : {}",
        expected_total
    );

    println!(
        "Headers received : {}",
        total
    );

    println!(
        "Final height     : {}",
        height
    );

    println!(
        "Final hash       : {}",
        locator
    );

    println!();

    if height == TARGET_HEIGHT
        && total as u64 == expected_total
    {
        println!(
            "TARGET REACHED"
        );

        println!(
            "{} consecutive Bitcoin headers verified.",
            total
        );

        println!(
            "Direct Bitcoin P2P acquisition SUCCESS."
        );
    } else {
        println!(
            "TARGET NOT REACHED"
        );
    }

    println!();

    println!(
        "DATASET SAFETY"
    );

    println!(
        "=============="
    );

    println!(
        "bitcoin-blocks.csv was NOT modified."
    );

    println!(
        "Frozen discovery dataset remains unchanged."
    );

    Ok(())
}
