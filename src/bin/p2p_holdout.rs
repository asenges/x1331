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

use csv::{Reader, Writer};
use rand::RngExt;

use std::{
    collections::HashSet,
    error::Error,
    fs,
    io::{Cursor, Read, Write},
    net::{SocketAddr, TcpStream, ToSocketAddrs},
    path::Path,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

const DISCOVERY: &str = "data/bitcoin-blocks.csv";
const OUTPUT: &str = "data/bitcoin-holdout-p2p.csv";

const ANCHOR_HEIGHT: u64 = 890_222;
const START_HEIGHT: u64 = 890_223;
const TARGET_HEIGHT: u64 = 899_999;

const EXPECTED_ROWS: u64 =
    TARGET_HEIGHT - START_HEIGHT + 1;

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
// OUTPUT ROW
// ============================================================

#[derive(Debug)]
struct HoldoutRow {
    height: u64,
    block_hash: String,
    header_hex: String,

    version: i32,
    previous_block_hash: String,
    merkle_root: String,

    timestamp: u32,
    bits: String,
    nonce: u32,

    x1331: [u8; 11],

    verified: bool,
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
// FROZEN DISCOVERY ANCHOR
// ============================================================

fn load_anchor(
) -> Result<BlockHash, Box<dyn Error>> {
    let mut reader =
        Reader::from_path(DISCOVERY)?;

    let columns =
        reader.headers()?.clone();

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

    let mut found:
        Option<BlockHash> = None;

    for row in reader.records() {
        let row = row?;

        let height: u64 =
            row.get(height_idx)
                .ok_or("missing height")?
                .parse()?;

        if height != ANCHOR_HEIGHT {
            continue;
        }

        if row.get(verified_idx)
            != Some("true")
        {
            return Err(
                format!(
                    "anchor {} is not verified",
                    ANCHOR_HEIGHT
                )
                .into(),
            );
        }

        let hash: BlockHash =
            row.get(hash_idx)
                .ok_or("missing block hash")?
                .parse()?;

        found = Some(hash);
        break;
    }

    found.ok_or_else(|| {
        format!(
            "anchor {} not found in {}",
            ANCHOR_HEIGHT,
            DISCOVERY
        )
        .into()
    })
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

    // Prefer IPv4.
    peers.sort_by_key(
        |p| !p.is_ipv4()
    );

    if peers.is_empty() {
        return Err(
            "DNS seed returned no peers".into()
        );
    }

    Ok(peers)
}


// ============================================================
// CONNECT TO FIRST WORKING PEER
// ============================================================

fn connect_peer(
) -> Result<(TcpStream, SocketAddr), Box<dyn Error>> {
    let peers =
        resolve_peers()?;

    for peer in peers {
        println!(
            "Trying peer   : {}",
            peer
        );

        match TcpStream::connect_timeout(
            &peer,
            Duration::from_secs(8),
        ) {
            Ok(stream) => {
                stream.set_read_timeout(
                    Some(
                        Duration::from_secs(30)
                    )
                )?;

                stream.set_write_timeout(
                    Some(
                        Duration::from_secs(30)
                    )
                )?;

                return Ok(
                    (stream, peer)
                );
            }

            Err(e) => {
                println!(
                    "  connect failed: {}",
                    e
                );
            }
        }
    }

    Err(
        "could not connect to any DNS-seed peer"
            .into()
    )
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

    let bytes =
        serialize(&message);

    stream.write_all(&bytes)?;
    stream.flush()?;

    Ok(())
}


// ============================================================
// READ BITCOIN P2P FRAME
// ============================================================

fn read_frame(
    stream: &mut TcpStream,
) -> Result<P2pFrame, Box<dyn Error>> {
    let mut envelope =
        [0u8; 24];

    stream.read_exact(
        &mut envelope
    )?;

    if envelope[0..4]
        != Magic::BITCOIN.to_bytes()
    {
        return Err(
            "invalid Bitcoin mainnet magic"
                .into()
        );
    }

    let command_bytes =
        &envelope[4..16];

    let end =
        command_bytes
            .iter()
            .position(|b| *b == 0)
            .unwrap_or(
                command_bytes.len()
            );

    let command =
        std::str::from_utf8(
            &command_bytes[..end]
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
        stream.read_exact(
            &mut payload
        )?;
    }

    let checksum =
        sha256d::Hash::hash(
            &payload
        )
        .to_byte_array();

    if checksum[0..4]
        != envelope[20..24]
    {
        return Err(
            format!(
                "checksum failure: {}",
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

    reader.read_exact(
        &mut first
    )?;

    match first[0] {
        0x00..=0xfc => {
            Ok(first[0] as u64)
        }

        0xfd => {
            let mut b = [0u8; 2];
            reader.read_exact(&mut b)?;

            Ok(
                u16::from_le_bytes(b)
                    as u64
            )
        }

        0xfe => {
            let mut b = [0u8; 4];
            reader.read_exact(&mut b)?;

            Ok(
                u32::from_le_bytes(b)
                    as u64
            )
        }

        0xff => {
            let mut b = [0u8; 8];
            reader.read_exact(&mut b)?;

            Ok(
                u64::from_le_bytes(b)
            )
        }
    }
}


// ============================================================
// DECODE HEADERS MESSAGE
// ============================================================

fn decode_headers_payload(
    payload: &[u8],
) -> Result<Vec<Header>, Box<dyn Error>> {
    let mut cursor =
        Cursor::new(payload);

    let count =
        read_compact_size(
            &mut cursor
        )?;

    if count
        > MAX_HEADERS_PER_BATCH as u64
    {
        return Err(
            format!(
                "peer returned {} headers",
                count
            )
            .into()
        );
    }

    let mut headers =
        Vec::with_capacity(
            count as usize
        );

    for i in 0..count {
        let mut raw =
            [0u8; 80];

        cursor.read_exact(
            &mut raw
        )?;

        let header: Header =
            deserialize(&raw)?;

        let tx_count =
            read_compact_size(
                &mut cursor
            )?;

        if tx_count != 0 {
            return Err(
                format!(
                    "headers[{}] tx_count={} expected 0",
                    i,
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
            format!(
                "headers payload not fully consumed: {} / {}",
                cursor.position(),
                payload.len()
            )
            .into()
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
            services
        );

    let local:
        SocketAddr =
        "0.0.0.0:0".parse()?;

    let sender =
        Address::new(
            &local,
            services
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
            "/X1331:0.18/".to_string(),
            0,
        );

    send_message(
        stream,
        NetworkMessage::Version(
            version
        ),
    )?;

    let mut version_ok = false;
    let mut verack_ok = false;

    while !(version_ok && verack_ok) {
        let frame =
            read_frame(stream)?;

        match frame.command.as_str() {
            "version" => {
                let v:
                    VersionMessage =
                    deserialize(
                        &frame.payload
                    )?;

                println!(
                    "Peer version  : {}",
                    v.version
                );

                println!(
                    "Peer software : {}",
                    v.user_agent
                );

                println!(
                    "Peer height   : {}",
                    v.start_height
                );

                version_ok = true;

                send_message(
                    stream,
                    NetworkMessage::Verack,
                )?;
            }

            "verack" => {
                verack_ok = true;
            }

            "ping" => {
                let n: u64 =
                    deserialize(
                        &frame.payload
                    )?;

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
// GET HEADERS
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
        NetworkMessage::GetHeaders(
            request
        ),
    )?;

    loop {
        let frame =
            read_frame(stream)?;

        match frame.command.as_str() {
            "headers" => {
                return
                    decode_headers_payload(
                        &frame.payload
                    );
            }

            "ping" => {
                let n: u64 =
                    deserialize(
                        &frame.payload
                    )?;

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
// X1331 NONCE STATES
// ============================================================
//
// 32-bit nonce:
//
// L1  bits 31..29
// L2  bits 28..26
// ...
// L10 bits  4..2
// L11 bits  1..0
//
// IMPORTANT:
// L11 is a 2-bit terminal state: 0..3.
//
// ============================================================

fn x1331_states(
    nonce: u32,
) -> [u8; 11] {
    let mut states =
        [0u8; 11];

    for level in 0..10 {
        let shift =
            29 - level * 3;

        states[level] =
            ((nonce >> shift) & 0b111)
                as u8;
    }

    states[10] =
        (nonce & 0b11) as u8;

    states
}


// ============================================================
// CONVERT HEADER TO DATASET ROW
// ============================================================

fn make_row(
    height: u64,
    header: &Header,
) -> Result<HoldoutRow, Box<dyn Error>> {
    let raw =
        serialize(header);

    if raw.len() != 80 {
        return Err(
            format!(
                "header {} serialized to {} bytes instead of 80",
                height,
                raw.len()
            )
            .into()
        );
    }

    // Local PoW verification.
    header
        .validate_pow(
            header.target()
        )
        .map_err(|e| {
            format!(
                "PoW failure at {}: {}",
                height,
                e
            )
        })?;

    let states =
        x1331_states(
            header.nonce
        );

    Ok(HoldoutRow {
        height,

        block_hash:
            header
                .block_hash()
                .to_string(),

        header_hex:
            hex::encode(raw),

        version:
            header
                .version
                .to_consensus(),

        previous_block_hash:
            header
                .prev_blockhash
                .to_string(),

        merkle_root:
            header
                .merkle_root
                .to_string(),

        timestamp:
            header.time,

        bits:
            format!(
                "{:08x}",
                header.bits.to_consensus()
            ),

        nonce:
            header.nonce,

        x1331:
            states,

        verified:
            true,
    })
}


// ============================================================
// WRITE CSV HEADER
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
// WRITE ROW
// ============================================================

fn write_row(
    writer: &mut Writer<std::fs::File>,
    row: &HoldoutRow,
) -> Result<(), Box<dyn Error>> {
    let record =
        vec![
            row.height.to_string(),
            row.block_hash.clone(),
            row.header_hex.clone(),

            row.version.to_string(),

            row.previous_block_hash.clone(),
            row.merkle_root.clone(),

            row.timestamp.to_string(),
            row.bits.clone(),
            row.nonce.to_string(),

            row.x1331[0].to_string(),
            row.x1331[1].to_string(),
            row.x1331[2].to_string(),
            row.x1331[3].to_string(),
            row.x1331[4].to_string(),
            row.x1331[5].to_string(),
            row.x1331[6].to_string(),
            row.x1331[7].to_string(),
            row.x1331[8].to_string(),
            row.x1331[9].to_string(),
            row.x1331[10].to_string(),

            row.verified.to_string(),
        ];

    writer.write_record(record)?;

    Ok(())
}


// ============================================================
// VALIDATE FINAL CSV
// ============================================================

fn validate_output(
) -> Result<(), Box<dyn Error>> {
    let mut reader =
        Reader::from_path(OUTPUT)?;

    let columns =
        reader.headers()?.clone();

    let height_idx =
        columns.iter()
            .position(|x| x == "height")
            .ok_or("missing height")?;

    let hash_idx =
        columns.iter()
            .position(|x| x == "block_hash")
            .ok_or("missing block_hash")?;

    let verified_idx =
        columns.iter()
            .position(|x| x == "verified")
            .ok_or("missing verified")?;

    let mut expected =
        START_HEIGHT;

    let mut rows =
        0u64;

    let mut hashes =
        HashSet::new();

    for record in reader.records() {
        let record =
            record?;

        let height: u64 =
            record
                .get(height_idx)
                .ok_or("missing height value")?
                .parse()?;

        if height != expected {
            return Err(
                format!(
                    "height gap: expected {}, found {}",
                    expected,
                    height
                )
                .into()
            );
        }

        if record.get(verified_idx)
            != Some("true")
        {
            return Err(
                format!(
                    "unverified row {}",
                    height
                )
                .into()
            );
        }

        let hash =
            record
                .get(hash_idx)
                .ok_or("missing hash")?
                .to_string();

        if !hashes.insert(hash) {
            return Err(
                format!(
                    "duplicate block hash at {}",
                    height
                )
                .into()
            );
        }

        expected += 1;
        rows += 1;
    }

    if rows != EXPECTED_ROWS {
        return Err(
            format!(
                "expected {} rows, found {}",
                EXPECTED_ROWS,
                rows
            )
            .into()
        );
    }

    if expected - 1 != TARGET_HEIGHT {
        return Err(
            format!(
                "last height incorrect: {}",
                expected - 1
            )
            .into()
        );
    }

    Ok(())
}


// ============================================================
// MAIN
// ============================================================

fn main(
) -> Result<(), Box<dyn Error>> {
    println!(
        "X1331 v0.18 P2P Holdout Builder"
    );

    println!(
        "================================"
    );

    println!(
        "Discovery : {}",
        DISCOVERY
    );

    println!(
        "Output    : {}",
        OUTPUT
    );

    println!(
        "Range     : {} .. {}",
        START_HEIGHT,
        TARGET_HEIGHT
    );

    println!(
        "Rows      : {}",
        EXPECTED_ROWS
    );

    println!();

    // --------------------------------------------------------
    // Refuse to overwrite an existing holdout.
    // --------------------------------------------------------

    if Path::new(OUTPUT).exists() {
        return Err(
            format!(
                "{} already exists; refusing to overwrite",
                OUTPUT
            )
            .into()
        );
    }

    // --------------------------------------------------------
    // Anchor from frozen discovery dataset.
    // --------------------------------------------------------

    let anchor_hash =
        load_anchor()?;

    println!(
        "Anchor    : {}",
        ANCHOR_HEIGHT
    );

    println!(
        "Hash      : {}",
        anchor_hash
    );

    // --------------------------------------------------------
    // Connect.
    // --------------------------------------------------------

    let (
        mut stream,
        peer,
    ) = connect_peer()?;

    println!(
        "Connected : {}",
        peer
    );

    handshake(
        &mut stream,
        peer
    )?;

    println!(
        "Handshake : OK"
    );

    println!();

    // --------------------------------------------------------
    // Temporary output.
    //
    // We only rename to final OUTPUT after complete validation.
    // --------------------------------------------------------

    let tmp =
        format!(
            "{}.tmp",
            OUTPUT
        );

    if Path::new(&tmp).exists() {
        fs::remove_file(
            &tmp
        )?;
    }

    let mut writer =
        Writer::from_path(
            &tmp
        )?;

    write_csv_header(
        &mut writer
    )?;

    let mut locator =
        anchor_hash;

    let mut height =
        ANCHOR_HEIGHT;

    let mut total =
        0u64;

    // --------------------------------------------------------
    // Download and write.
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
            // ------------------------------------------------
            // Chain continuity.
            // ------------------------------------------------

            if header.prev_blockhash
                != expected_prev
            {
                return Err(
                    format!(
                        "broken chain before height {}",
                        height + 1
                    )
                    .into()
                );
            }

            height += 1;

            let row =
                make_row(
                    height,
                    header
                )?;

            write_row(
                &mut writer,
                &row
            )?;

            expected_prev =
                header.block_hash();

            locator =
                expected_prev;

            total += 1;
        }

        writer.flush()?;

        println!(
            "Saved {:4} verified headers: {} .. {} | total {}",
            usable,
            batch_start,
            batch_end,
            total
        );

        if height >= TARGET_HEIGHT {
            break;
        }

        if headers.len()
            < MAX_HEADERS_PER_BATCH
        {
            return Err(
                format!(
                    "short headers response at height {}",
                    height
                )
                .into()
            );
        }
    }

    drop(writer);

    // --------------------------------------------------------
    // Move temporary file into final location.
    // --------------------------------------------------------

    fs::rename(
        &tmp,
        OUTPUT
    )?;

    // --------------------------------------------------------
    // Structural validation.
    // --------------------------------------------------------

    validate_output()?;

    println!();

    println!(
        "HOLDOUT BUILD COMPLETE"
    );

    println!(
        "======================"
    );

    println!(
        "Rows         : {}",
        total
    );

    println!(
        "First height : {}",
        START_HEIGHT
    );

    println!(
        "Last height  : {}",
        TARGET_HEIGHT
    );

    println!(
        "Final hash   : {}",
        locator
    );

    println!(
        "Output       : {}",
        OUTPUT
    );

    println!();

    println!(
        "VALIDATION PASSED"
    );

    println!(
        "- consecutive heights"
    );

    println!(
        "- chain linkage"
    );

    println!(
        "- local PoW"
    );

    println!(
        "- 80-byte serialized headers"
    );

    println!(
        "- unique block hashes"
    );

    println!(
        "- {} verified rows",
        EXPECTED_ROWS
    );

    println!();

    println!(
        "IMPORTANT: no bit7/H1 statistics were calculated."
    );

    println!(
        "The holdout remains statistically unopened."
    );

    Ok(())
}
