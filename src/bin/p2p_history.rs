use bitcoin::{
    block::Header,
    blockdata::constants::genesis_block,
    consensus::{deserialize, encode::serialize},
    hashes::{sha256d, Hash},
    p2p::{
        address::Address,
        message::{NetworkMessage, RawNetworkMessage},
        message_blockdata::GetHeadersMessage,
        message_network::VersionMessage,
        Magic, ServiceFlags,
    },
    BlockHash, Network,
};

use csv::{Reader, Writer, WriterBuilder};
use rand::RngExt;

use std::{
    collections::HashSet,
    error::Error,
    fs::{self, OpenOptions},
    io::{Cursor, Read, Write},
    net::{IpAddr, SocketAddr, TcpStream, ToSocketAddrs},
    path::Path,
    str::FromStr,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

const STORE_START: u64 = 700_000;
const TARGET_HEIGHT: u64 = 899_999;

const OUTPUT: &str =
    "data/bitcoin-history-700000-899999.csv";

const CHECKPOINT: &str =
    "data/v020-p2p-checkpoint.csv";

const CHECKPOINT_TMP: &str =
    "data/v020-p2p-checkpoint.tmp";

const MAX_HEADERS_PER_BATCH: usize = 2000;
const MAX_P2P_PAYLOAD: usize = 32 * 1024 * 1024;

const CONNECT_TIMEOUT: u64 = 8;
const IO_TIMEOUT: u64 = 30;

const DNS_SEEDS: &[&str] = &[
    "seed.bitcoin.sipa.be",
    "dnsseed.bluematt.me",
    "seed.bitcoin.jonasschnelli.ch",
    "seed.btc.petertodd.net",
    "seed.bitcoin.sprovoost.nl",
    "dnsseed.emzy.de",
    "seed.bitcoin.wiz.biz",
];

struct P2pFrame {
    command: String,
    payload: Vec<u8>,
}

#[derive(Clone, Copy)]
struct SyncState {
    height: u64,
    hash: BlockHash,
}

struct PeerConnection {
    stream: TcpStream,
    address: SocketAddr,
    advertised_height: i32,
    user_agent: String,
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

fn discover_peers() -> Vec<SocketAddr> {
    let mut unique =
        HashSet::<SocketAddr>::new();

    println!("Discovering Bitcoin peers...");

    for seed in DNS_SEEDS {
        match (*seed, 8333).to_socket_addrs() {
            Ok(addresses) => {
                let mut count = 0usize;

                for addr in addresses {
                    if unique.insert(addr) {
                        count += 1;
                    }
                }

                println!(
                    "  {:35} +{}",
                    seed,
                    count
                );
            }

            Err(e) => {
                println!(
                    "  {:35} DNS error: {}",
                    seed,
                    e
                );
            }
        }
    }

    let mut peers: Vec<_> =
        unique.into_iter().collect();

    peers.sort_by_key(|p| !p.is_ipv4());

    println!(
        "Peer pool: {} unique addresses",
        peers.len()
    );

    peers
}


// ============================================================
// SEND MESSAGE
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
// READ FRAME
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
                "checksum mismatch for {}",
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
        0x00..=0xfc =>
            Ok(first[0] as u64),

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
                    "header {} has tx_count {}",
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
) -> Result<(i32, String), Box<dyn Error>> {
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
            "/X1331:0.20.1/".to_string(),
            0,
        );

    send_message(
        stream,
        NetworkMessage::Version(version),
    )?;

    let mut got_version = false;
    let mut got_verack = false;

    let mut advertised_height = 0i32;
    let mut user_agent =
        String::from("unknown");

    while !(got_version && got_verack) {
        let frame =
            read_frame(stream)?;

        match frame.command.as_str() {
            "version" => {
                let v: VersionMessage =
                    deserialize(&frame.payload)?;

                advertised_height =
                    v.start_height;

                user_agent =
                    v.user_agent.clone();

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

    Ok((
        advertised_height,
        user_agent,
    ))
}


// ============================================================
// CONNECT NEXT PEER
// ============================================================

fn connect_next_peer(
    peers: &[SocketAddr],
    cursor: &mut usize,
    failed: &mut HashSet<SocketAddr>,
) -> Result<PeerConnection, Box<dyn Error>> {
    while *cursor < peers.len() {
        let peer = peers[*cursor];
        *cursor += 1;

        if failed.contains(&peer) {
            continue;
        }

        println!(
            "Connecting peer {}/{}: {}",
            *cursor,
            peers.len(),
            peer
        );

        let mut stream =
            match TcpStream::connect_timeout(
                &peer,
                Duration::from_secs(
                    CONNECT_TIMEOUT
                ),
            ) {
                Ok(s) => s,

                Err(e) => {
                    println!(
                        "  connect failed: {}",
                        e
                    );

                    failed.insert(peer);
                    continue;
                }
            };

        stream.set_read_timeout(
            Some(
                Duration::from_secs(IO_TIMEOUT)
            )
        )?;

        stream.set_write_timeout(
            Some(
                Duration::from_secs(IO_TIMEOUT)
            )
        )?;

        match handshake(
            &mut stream,
            peer
        ) {
            Ok((height, agent)) => {
                println!(
                    "  READY | {} | advertised height {}",
                    agent,
                    height
                );

                return Ok(PeerConnection {
                    stream,
                    address: peer,
                    advertised_height: height,
                    user_agent: agent,
                });
            }

            Err(e) => {
                println!(
                    "  handshake failed: {}",
                    e
                );

                failed.insert(peer);
            }
        }
    }

    Err(
        "peer pool exhausted".into()
    )
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
// CHECKPOINT
// ============================================================

fn save_checkpoint(
    state: SyncState,
) -> Result<(), Box<dyn Error>> {
    {
        let mut w =
            Writer::from_path(CHECKPOINT_TMP)?;

        w.write_record([
            "height",
            "block_hash",
        ])?;

        w.write_record([
            state.height.to_string(),
            state.hash.to_string(),
        ])?;

        w.flush()?;
    }

    fs::rename(
        CHECKPOINT_TMP,
        CHECKPOINT
    )?;

    Ok(())
}


fn load_checkpoint(
) -> Result<Option<SyncState>, Box<dyn Error>> {
    if !Path::new(CHECKPOINT).exists() {
        return Ok(None);
    }

    let mut reader =
        Reader::from_path(CHECKPOINT)?;

    let row =
        match reader.records().next() {
            Some(r) => r?,
            None => return Ok(None),
        };

    let height: u64 =
        row.get(0)
            .ok_or("checkpoint height missing")?
            .parse()?;

    let hash =
        BlockHash::from_str(
            row.get(1)
                .ok_or(
                    "checkpoint hash missing"
                )?
        )?;

    Ok(Some(SyncState {
        height,
        hash,
    }))
}


// ============================================================
// X1331
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

        out[level] =
            format!(
                "{:03b}",
                (nonce >> shift) & 7
            );
    }

    out[10] =
        format!(
            "{:03b}",
            nonce & 3
        );

    out
}


// ============================================================
// CSV
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


fn write_header(
    writer: &mut Writer<std::fs::File>,
    height: u64,
    header: &Header,
) -> Result<(), Box<dyn Error>> {
    let raw = serialize(header);

    if raw.len() != 80 {
        return Err(
            format!(
                "height {} serialized to {} bytes",
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
// OPEN OUTPUT
// ============================================================

fn open_output(
    state: SyncState,
) -> Result<
    Writer<std::fs::File>,
    Box<dyn Error>
> {
    let exists =
        Path::new(OUTPUT).exists();

    if state.height >= STORE_START
        && !exists
    {
        return Err(
            format!(
                "checkpoint is at {}, but {} does not exist; refusing to skip stored history",
                state.height,
                OUTPUT
            )
            .into()
        );
    }

    if !exists {
        let file =
            OpenOptions::new()
                .create(true)
                .append(true)
                .open(OUTPUT)?;

        let mut writer =
            WriterBuilder::new()
                .has_headers(false)
                .from_writer(file);

        write_csv_header(
            &mut writer
        )?;

        writer.flush()?;

        return Ok(writer);
    }

    let file =
        OpenOptions::new()
            .append(true)
            .open(OUTPUT)?;

    Ok(
        WriterBuilder::new()
            .has_headers(false)
            .from_writer(file)
    )
}


// ============================================================
// MAIN
// ============================================================

fn main(
) -> Result<(), Box<dyn Error>> {
    println!(
        "X1331 v0.20.1 Historical P2P Engine"
    );

    println!(
        "===================================="
    );

    println!(
        "Target       : {}",
        TARGET_HEIGHT
    );

    println!(
        "Store range  : {} .. {}",
        STORE_START,
        TARGET_HEIGHT
    );

    println!(
        "Output       : {}",
        OUTPUT
    );

    println!(
        "Checkpoint   : {}",
        CHECKPOINT
    );

    println!();

    // --------------------------------------------------------
    // Initial state
    // --------------------------------------------------------

    let genesis =
        genesis_block(Network::Bitcoin);

    let genesis_hash =
        genesis.header.block_hash();

    let mut state =
        match load_checkpoint()? {
            Some(s) => {
                println!(
                    "RESUME: height {} | {}",
                    s.height,
                    s.hash
                );

                s
            }

            None => {
                println!(
                    "START: genesis | {}",
                    genesis_hash
                );

                SyncState {
                    height: 0,
                    hash: genesis_hash,
                }
            }
        };

    if state.height >= TARGET_HEIGHT {
        println!("Target already reached.");
        return Ok(());
    }

    // --------------------------------------------------------
    // Output
    // --------------------------------------------------------

    let mut writer =
        open_output(state)?;

    // --------------------------------------------------------
    // Peer pool
    // --------------------------------------------------------

    let peers =
        discover_peers();

    if peers.is_empty() {
        return Err(
            "no peers discovered".into()
        );
    }

    let mut peer_cursor = 0usize;

    let mut failed =
        HashSet::<SocketAddr>::new();

    let mut connection:
        Option<PeerConnection> = None;

    let mut session_headers =
        0u64;

    // --------------------------------------------------------
    // Sync
    // --------------------------------------------------------

    while state.height < TARGET_HEIGHT {
        if connection.is_none() {
            let p =
                connect_next_peer(
                    &peers,
                    &mut peer_cursor,
                    &mut failed,
                )?;

            println!(
                "SYNC peer {} | locator height {}",
                p.address,
                state.height
            );

            connection = Some(p);
            session_headers = 0;
        }

        let result = {
            let p =
                connection
                    .as_mut()
                    .unwrap();

            request_headers(
                &mut p.stream,
                state.hash
            )
        };

        let headers =
            match result {
                Ok(h) => h,

                Err(e) => {
                    let p =
                        connection
                            .take()
                            .unwrap();

                    println!(
                        "PEER LOST {} after {} headers: {}",
                        p.address,
                        session_headers,
                        e
                    );

                    failed.insert(
                        p.address
                    );

                    println!(
                        "Failover from height {}...",
                        state.height
                    );

                    continue;
                }
            };

        if headers.is_empty() {
            let p =
                connection
                    .take()
                    .unwrap();

            println!(
                "Peer {} returned zero headers; rotating.",
                p.address
            );

            failed.insert(
                p.address
            );

            continue;
        }

        let remaining =
            TARGET_HEIGHT - state.height;

        let usable =
            std::cmp::min(
                headers.len(),
                remaining as usize,
            );

        let batch_start =
            state.height + 1;

        let mut expected_prev =
            state.hash;

        let mut batch_last =
            state;

        // ----------------------------------------------------
        // VERIFY ENTIRE USABLE BATCH FIRST
        // ----------------------------------------------------

        for (
            index,
            header
        ) in headers
            .iter()
            .take(usable)
            .enumerate()
        {
            let height =
                state.height
                    + index as u64
                    + 1;

            if header.prev_blockhash
                != expected_prev
            {
                return Err(
                    format!(
                        "CHAIN FAILURE at {} from peer {}",
                        height,
                        connection
                            .as_ref()
                            .unwrap()
                            .address
                    )
                    .into()
                );
            }

            header
                .validate_pow(
                    header.target()
                )
                .map_err(|e| {
                    format!(
                        "PoW FAILURE at {}: {}",
                        height,
                        e
                    )
                })?;

            expected_prev =
                header.block_hash();

            batch_last =
                SyncState {
                    height,
                    hash: expected_prev,
                };
        }

        // ----------------------------------------------------
        // WRITE STORED RANGE
        // ----------------------------------------------------

        for (
            index,
            header
        ) in headers
            .iter()
            .take(usable)
            .enumerate()
        {
            let height =
                state.height
                    + index as u64
                    + 1;

            if height >= STORE_START {
                write_header(
                    &mut writer,
                    height,
                    header
                )?;
            }
        }

        // CSV durable before checkpoint advances.
        writer.flush()?;

        // ----------------------------------------------------
        // COMMIT STATE
        // ----------------------------------------------------

        state =
            batch_last;

        session_headers +=
            usable as u64;

        save_checkpoint(
            state
        )?;

        println!(
            "OK {:6} .. {:6} | height {:6} | stored {:6} | peer {}",
            batch_start,
            state.height,
            state.height,
            if state.height >= STORE_START {
                state.height - STORE_START + 1
            } else {
                0
            },
            connection
                .as_ref()
                .unwrap()
                .address
        );

        if state.height >= TARGET_HEIGHT {
            break;
        }

        // A short response before target can mean
        // peer tip/limited history. Rotate safely.
        if headers.len()
            < MAX_HEADERS_PER_BATCH
        {
            let p =
                connection
                    .take()
                    .unwrap();

            println!(
                "Peer {} returned short batch {}; rotating.",
                p.address,
                headers.len()
            );

            failed.insert(
                p.address
            );
        }
    }

    writer.flush()?;

    println!();

    println!(
        "HISTORICAL SYNC COMPLETE"
    );

    println!(
        "========================"
    );

    println!(
        "Final height : {}",
        state.height
    );

    println!(
        "Final hash   : {}",
        state.hash
    );

    println!(
        "Expected hash: 0000000000000000000196400396be46d0816dc462df4c3450972f589f4d7d24"
    );

    println!(
        "Stored rows  : {}",
        state.height - STORE_START + 1
    );

    println!(
        "Output       : {}",
        OUTPUT
    );

    println!(
        "Checkpoint   : {}",
        CHECKPOINT
    );

    println!();

    println!(
        "No H1/bit7 analysis performed."
    );

    Ok(())
}
