use bitcoin::{
    block::Header,
    consensus::{deserialize, encode::serialize},
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

use csv::Writer;
use rand::RngExt;

use std::{
    error::Error,
    fs,
    io::{Cursor, Read, Write},
    net::{SocketAddr, TcpStream, ToSocketAddrs},
    path::Path,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

const STORE_START: u64 = 700_000;
const TARGET_HEIGHT: u64 = 899_999;

const OUTPUT: &str =
    "data/bitcoin-history-700000-899999.csv";

const TMP_OUTPUT: &str =
    "data/bitcoin-history-700000-899999.tmp.csv";

const MAX_HEADERS_PER_BATCH: usize = 2000;
const MAX_P2P_PAYLOAD: usize = 32 * 1024 * 1024;


// ============================================================
// P2P FRAME
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
// PEER DISCOVERY
// ============================================================

fn resolve_peers(
) -> Result<Vec<SocketAddr>, Box<dyn Error>> {
    let addresses =
        ("seed.bitcoin.sipa.be", 8333)
            .to_socket_addrs()?;

    let mut peers:
        Vec<SocketAddr> =
        addresses.collect();

    peers.sort_by_key(|p| !p.is_ipv4());

    if peers.is_empty() {
        return Err(
            "DNS seed returned no peers".into()
        );
    }

    Ok(peers)
}


fn connect_peer(
) -> Result<(TcpStream, SocketAddr), Box<dyn Error>> {
    let peers = resolve_peers()?;

    for peer in peers {
        println!("Trying peer: {}", peer);

        match TcpStream::connect_timeout(
            &peer,
            Duration::from_secs(10),
        ) {
            Ok(stream) => {
                stream.set_read_timeout(
                    Some(Duration::from_secs(30)),
                )?;

                stream.set_write_timeout(
                    Some(Duration::from_secs(30)),
                )?;

                return Ok((stream, peer));
            }

            Err(e) => {
                println!(
                    "  connection failed: {}",
                    e
                );
            }
        }
    }

    Err("could not connect to a Bitcoin peer".into())
}


// ============================================================
// SEND
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
// RAW P2P FRAME
// ============================================================

fn read_frame(
    stream: &mut TcpStream,
) -> Result<P2pFrame, Box<dyn Error>> {
    let mut envelope = [0u8; 24];

    stream.read_exact(&mut envelope)?;

    if envelope[0..4]
        != Magic::BITCOIN.to_bytes()
    {
        return Err(
            "invalid Bitcoin mainnet magic".into()
        );
    }

    let command_bytes =
        &envelope[4..16];

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
                "unreasonable payload: {}",
                payload_len
            )
            .into()
        );
    }

    let mut payload =
        vec![0u8; payload_len];

    if payload_len > 0 {
        stream.read_exact(&mut payload)?;
    }

    let digest =
        sha256d::Hash::hash(&payload)
            .to_byte_array();

    if digest[0..4] != envelope[20..24] {
        return Err(
            format!(
                "checksum mismatch: {}",
                command
            )
            .into()
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
        0x00..=0xfc => Ok(first[0] as u64),

        0xfd => {
            let mut b = [0u8; 2];
            reader.read_exact(&mut b)?;
            Ok(u16::from_le_bytes(b) as u64)
        }

        0xfe => {
            let mut b = [0u8; 4];
            reader.read_exact(&mut b)?;
            Ok(u32::from_le_bytes(b) as u64)
        }

        0xff => {
            let mut b = [0u8; 8];
            reader.read_exact(&mut b)?;
            Ok(u64::from_le_bytes(b))
        }
    }
}


// ============================================================
// HEADERS DECODER
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
                "peer returned {} headers",
                count
            )
            .into()
        );
    }

    let mut headers =
        Vec::with_capacity(count as usize);

    for index in 0..count {
        let mut raw = [0u8; 80];

        cursor.read_exact(&mut raw)?;

        let header: Header =
            deserialize(&raw)?;

        let tx_count =
            read_compact_size(&mut cursor)?;

        if tx_count != 0 {
            return Err(
                format!(
                    "header {} tx_count={} expected 0",
                    index,
                    tx_count
                )
                .into()
            );
        }

        headers.push(header);
    }

    if cursor.position() as usize
        != payload.len()
    {
        return Err(
            "headers payload not fully consumed".into()
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
        Address::new(&peer, services);

    let local: SocketAddr =
        "0.0.0.0:0".parse()?;

    let sender =
        Address::new(&local, services);

    let nonce: u64 =
        rand::rng().random();

    let version =
        VersionMessage::new(
            services,
            unix_time(),
            receiver,
            sender,
            nonce,
            "/X1331:0.20/".to_string(),
            0,
        );

    send_message(
        stream,
        NetworkMessage::Version(version),
    )?;

    let mut got_version = false;
    let mut got_verack = false;

    while !(got_version && got_verack) {
        let frame = read_frame(stream)?;

        match frame.command.as_str() {
            "version" => {
                let v: VersionMessage =
                    deserialize(&frame.payload)?;

                println!(
                    "Peer: {} | height {}",
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
                got_verack = true;
            }

            "ping" => {
                let n: u64 =
                    deserialize(&frame.payload)?;

                send_message(
                    stream,
                    NetworkMessage::Pong(n),
                )?;
            }

            _ => {}
        }
    }

    Ok(())
}


// ============================================================
// GETHEADERS
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
        let frame = read_frame(stream)?;

        match frame.command.as_str() {
            "headers" => {
                return decode_headers_payload(
                    &frame.payload
                );
            }

            "ping" => {
                let n: u64 =
                    deserialize(&frame.payload)?;

                send_message(
                    stream,
                    NetworkMessage::Pong(n),
                )?;
            }

            _ => {}
        }
    }
}


// ============================================================
// X1331 STATES
// ============================================================

fn x1331_states(
    nonce: u32,
) -> [String; 11] {
    let mut out: [String; 11] =
        std::array::from_fn(
            |_| String::new()
        );

    for level in 0..10 {
        let shift =
            29 - level * 3;

        let state =
            (nonce >> shift) & 0b111;

        out[level] =
            format!("{:03b}", state);
    }

    out[10] =
        format!("{:03b}", nonce & 0b11);

    out
}


// ============================================================
// CSV HEADER
// ============================================================

fn write_csv_header(
    writer: &mut Writer<std::fs::File>,
) -> Result<(), Box<dyn Error>> {
    writer.write_record([
        "height",
        "block_hash",
        "header_hex",
        "version",
        "previous_block_hash",
        "merkle_root",
        "timestamp",
        "bits",
        "nonce",
        "x1331_l1",
        "x1331_l2",
        "x1331_l3",
        "x1331_l4",
        "x1331_l5",
        "x1331_l6",
        "x1331_l7",
        "x1331_l8",
        "x1331_l9",
        "x1331_l10",
        "x1331_l11",
        "verified",
    ])?;

    Ok(())
}


// ============================================================
// WRITE ONE STORED HEADER
// ============================================================

fn write_header(
    writer: &mut Writer<std::fs::File>,
    height: u64,
    header: &Header,
) -> Result<(), Box<dyn Error>> {
    let raw = serialize(header);

    if raw.len() != 80 {
        return Err(
            format!(
                "height {} serialized length {}",
                height,
                raw.len()
            )
            .into()
        );
    }

    let x =
        x1331_states(header.nonce);

    writer.write_record([
        height.to_string(),
        header.block_hash().to_string(),
        hex::encode(raw),

        header
            .version
            .to_consensus()
            .to_string(),

        header
            .prev_blockhash
            .to_string(),

        header
            .merkle_root
            .to_string(),

        header.time.to_string(),

        // IMPORTANT:
        // Keep compatibility with original dataset:
        // bits stored as DECIMAL.
        header
            .bits
            .to_consensus()
            .to_string(),

        header.nonce.to_string(),

        x[0].clone(),
        x[1].clone(),
        x[2].clone(),
        x[3].clone(),
        x[4].clone(),
        x[5].clone(),
        x[6].clone(),
        x[7].clone(),
        x[8].clone(),
        x[9].clone(),
        x[10].clone(),

        "true".to_string(),
    ])?;

    Ok(())
}


// ============================================================
// MAIN
// ============================================================

fn main(
) -> Result<(), Box<dyn Error>> {
    println!(
        "X1331 v0.20 Historical P2P Builder"
    );

    println!(
        "=================================="
    );

    println!(
        "Walk chain : genesis .. {}",
        TARGET_HEIGHT
    );

    println!(
        "Store      : {} .. {}",
        STORE_START,
        TARGET_HEIGHT
    );

    println!(
        "Rows       : {}",
        TARGET_HEIGHT - STORE_START + 1
    );

    println!(
        "Output     : {}",
        OUTPUT
    );

    println!();

    if Path::new(OUTPUT).exists() {
        return Err(
            format!(
                "{} already exists; refusing overwrite",
                OUTPUT
            )
            .into()
        );
    }

    if Path::new(TMP_OUTPUT).exists() {
        fs::remove_file(TMP_OUTPUT)?;
    }

    // --------------------------------------------------------
    // Bitcoin mainnet genesis.
    //
    // rust-bitcoin gives us the canonical genesis block.
    // We use its hash as the first getheaders locator.
    // Height starts at 0.
    // --------------------------------------------------------

    let genesis =
        bitcoin::blockdata::constants::genesis_block(
            bitcoin::Network::Bitcoin
        );

    let genesis_header =
        genesis.header;

    genesis_header
        .validate_pow(
            genesis_header.target()
        )
        .map_err(|e| {
            format!(
                "genesis PoW validation failed: {}",
                e
            )
        })?;

    let genesis_hash =
        genesis_header.block_hash();

    println!(
        "Genesis hash: {}",
        genesis_hash
    );

    let (
        mut stream,
        peer,
    ) = connect_peer()?;

    println!("Connected   : {}", peer);

    handshake(
        &mut stream,
        peer
    )?;

    println!("Handshake   : OK");
    println!();

    let mut writer =
        Writer::from_path(TMP_OUTPUT)?;

    write_csv_header(
        &mut writer
    )?;

    let mut height = 0u64;
    let mut locator =
        genesis_hash;

    let mut walked =
        0u64;

    let mut stored =
        0u64;

    // --------------------------------------------------------
    // Walk forward from genesis.
    // --------------------------------------------------------

    while height < TARGET_HEIGHT {
        let headers =
            request_headers(
                &mut stream,
                locator
            )?;

        if headers.is_empty() {
            return Err(
                format!(
                    "peer stopped at height {}",
                    height
                )
                .into()
            );
        }

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

        let mut expected_prev =
            locator;

        for header in
            headers.iter().take(usable)
        {
            let next_height =
                height + 1;

            // -----------------------------------------------
            // Chain linkage
            // -----------------------------------------------

            if header.prev_blockhash
                != expected_prev
            {
                return Err(
                    format!(
                        "broken chain at height {}",
                        next_height
                    )
                    .into()
                );
            }

            // -----------------------------------------------
            // Local PoW
            // -----------------------------------------------

            header
                .validate_pow(
                    header.target()
                )
                .map_err(|e| {
                    format!(
                        "invalid PoW at height {}: {}",
                        next_height,
                        e
                    )
                })?;

            height =
                next_height;

            walked += 1;

            if height >= STORE_START {
                write_header(
                    &mut writer,
                    height,
                    header
                )?;

                stored += 1;
            }

            expected_prev =
                header.block_hash();

            locator =
                expected_prev;
        }

        if stored > 0 {
            writer.flush()?;
        }

        // Print one progress line per 20k heights,
        // plus when storage begins and at target.
        if batch_end >= STORE_START
            || batch_end % 20_000 == 0
            || height == TARGET_HEIGHT
        {
            println!(
                "Verified {} .. {} | height {} | stored {}",
                batch_start,
                batch_end,
                height,
                stored
            );
        }

        if height >= TARGET_HEIGHT {
            break;
        }

        if headers.len()
            < MAX_HEADERS_PER_BATCH
        {
            return Err(
                format!(
                    "short headers response at height {}: {}",
                    height,
                    headers.len()
                )
                .into()
            );
        }
    }

    drop(writer);

    // --------------------------------------------------------
    // Final invariants
    // --------------------------------------------------------

    let expected_stored =
        TARGET_HEIGHT - STORE_START + 1;

    if height != TARGET_HEIGHT {
        return Err(
            format!(
                "wrong final height: {}",
                height
            )
            .into()
        );
    }

    if stored != expected_stored {
        return Err(
            format!(
                "expected {} stored rows, got {}",
                expected_stored,
                stored
            )
            .into()
        );
    }

    // Atomic-ish finalization:
    // only expose OUTPUT after complete run.
    fs::rename(
        TMP_OUTPUT,
        OUTPUT
    )?;

    println!();

    println!(
        "HISTORICAL DATASET COMPLETE"
    );

    println!(
        "==========================="
    );

    println!(
        "Headers walked : {}",
        walked
    );

    println!(
        "Stored rows    : {}",
        stored
    );

    println!(
        "Stored range   : {} .. {}",
        STORE_START,
        TARGET_HEIGHT
    );

    println!(
        "Final height   : {}",
        height
    );

    println!(
        "Final hash     : {}",
        locator
    );

    println!(
        "Output         : {}",
        OUTPUT
    );

    println!();

    println!(
        "No bit7 statistics were calculated."
    );

    println!(
        "Dataset is ready for v0.20 regime mapping."
    );

    Ok(())
}
