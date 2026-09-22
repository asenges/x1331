use std::fmt;

/// Bit ordering used when viewing authoritative raw bytes.
///
/// MSB-first means bit coordinate 0 of a byte is bit 7:
///
///   byte = abcdefgh
///          ^
///          bit coordinate 0
///
/// Raw bytes are never modified by this view.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BitOrder {
    MsbFirst,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum X1331State {
    S000 = 0b000,
    S001 = 0b001,
    S010 = 0b010,
    S011 = 0b011,
    S100 = 0b100,
    S101 = 0b101,
    S110 = 0b110,
    S111 = 0b111,
}

impl X1331State {
    pub const ALL: [Self; 8] = [
        Self::S000,
        Self::S001,
        Self::S010,
        Self::S011,
        Self::S100,
        Self::S101,
        Self::S110,
        Self::S111,
    ];

    pub fn from_u8(value: u8) -> Self {
        match value & 0b111 {
            0b000 => Self::S000,
            0b001 => Self::S001,
            0b010 => Self::S010,
            0b011 => Self::S011,
            0b100 => Self::S100,
            0b101 => Self::S101,
            0b110 => Self::S110,
            0b111 => Self::S111,
            _ => unreachable!(),
        }
    }

    pub fn value(self) -> u8 {
        self as u8
    }

    pub fn bits(self) -> [u8; 3] {
        let value = self.value();

        [(value >> 2) & 1, (value >> 1) & 1, value & 1]
    }

    /// Hamming weight gives the canonical X1331 1|3|3|1 layer.
    pub fn layer(self) -> u8 {
        self.value().count_ones() as u8
    }

    /// The three Boolean-cube neighbors obtained by flipping exactly one bit.
    pub fn neighbors(self) -> [Self; 3] {
        let value = self.value();

        [
            Self::from_u8(value ^ 0b100),
            Self::from_u8(value ^ 0b010),
            Self::from_u8(value ^ 0b001),
        ]
    }

    pub fn complement(self) -> Self {
        Self::from_u8((!self.value()) & 0b111)
    }

    pub fn hamming_distance(self, other: Self) -> u8 {
        (self.value() ^ other.value()).count_ones() as u8
    }
}

impl fmt::Display for X1331State {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:03b}", self.value())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct X1331Cell {
    /// Coordinate of the first raw bit represented by this cell.
    pub start_bit: usize,

    /// Three raw bits in matrix order.
    pub bits: [u8; 3],

    /// Exact state represented by those bits.
    pub state: X1331State,
}

impl X1331Cell {
    pub fn new(start_bit: usize, bits: [u8; 3]) -> Self {
        assert!(bits.iter().all(|bit| *bit <= 1));

        let value = (bits[0] << 2) | (bits[1] << 1) | bits[2];

        Self {
            start_bit,
            bits,
            state: X1331State::from_u8(value),
        }
    }

    pub fn end_bit_exclusive(&self) -> usize {
        self.start_bit + 3
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FigureTransition {
    pub from: X1331State,
    pub to: X1331State,
    pub xor: X1331State,
    pub hamming_distance: u8,
    pub from_layer: u8,
    pub to_layer: u8,
}

impl FigureTransition {
    pub fn new(from: X1331State, to: X1331State) -> Self {
        Self {
            from,
            to,
            xor: X1331State::from_u8(from.value() ^ to.value()),
            hamming_distance: from.hamming_distance(to),
            from_layer: from.layer(),
            to_layer: to.layer(),
        }
    }
}

/// Lossless binary substrate.
///
/// `raw` is authoritative. Cells and figures are derived views.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct X1331BitMatrix {
    raw: Vec<u8>,
    order: BitOrder,
}

impl X1331BitMatrix {
    pub fn from_bytes(raw: Vec<u8>) -> Self {
        Self {
            raw,
            order: BitOrder::MsbFirst,
        }
    }

    pub fn raw_bytes(&self) -> &[u8] {
        &self.raw
    }

    pub fn into_raw_bytes(self) -> Vec<u8> {
        self.raw
    }

    pub fn bit_order(&self) -> BitOrder {
        self.order
    }

    pub fn bit_len(&self) -> usize {
        self.raw.len() * 8
    }

    pub fn bit(&self, bit_index: usize) -> Option<u8> {
        if bit_index >= self.bit_len() {
            return None;
        }

        let byte_index = bit_index / 8;
        let within_byte = bit_index % 8;

        let shift = 7 - within_byte;

        Some((self.raw[byte_index] >> shift) & 1)
    }

    pub fn bits(&self, start_bit: usize, count: usize) -> Option<Vec<u8>> {
        let end = start_bit.checked_add(count)?;

        if end > self.bit_len() {
            return None;
        }

        let mut result = Vec::with_capacity(count);

        for index in start_bit..end {
            result.push(self.bit(index)?);
        }

        Some(result)
    }

    /// Read an exact 3-bit X1331 cell.
    ///
    /// Cells are allowed to begin at any raw bit coordinate. This is
    /// intentional: future figure experiments may use overlapping windows.
    pub fn cell_at(&self, start_bit: usize) -> Option<X1331Cell> {
        if start_bit.checked_add(3)? > self.bit_len() {
            return None;
        }

        let bits = [
            self.bit(start_bit)?,
            self.bit(start_bit + 1)?,
            self.bit(start_bit + 2)?,
        ];

        Some(X1331Cell::new(start_bit, bits))
    }

    /// Produce non-overlapping 3-bit cells from a chosen starting coordinate.
    ///
    /// Any final 1-2 bits that cannot form a complete cell are intentionally
    /// left in the authoritative raw matrix rather than padded or discarded.
    pub fn cells_from(&self, start_bit: usize) -> Vec<X1331Cell> {
        if start_bit >= self.bit_len() {
            return Vec::new();
        }

        let available = self.bit_len() - start_bit;
        let cell_count = available / 3;

        let mut cells = Vec::with_capacity(cell_count);

        for i in 0..cell_count {
            let coordinate = start_bit + i * 3;

            if let Some(cell) = self.cell_at(coordinate) {
                cells.push(cell);
            }
        }

        cells
    }

    pub fn transitions_from(&self, start_bit: usize) -> Vec<FigureTransition> {
        let cells = self.cells_from(start_bit);

        if cells.len() < 2 {
            return Vec::new();
        }

        cells
            .windows(2)
            .map(|pair| FigureTransition::new(pair[0].state, pair[1].state))
            .collect()
    }

    /// Reconstruct complete bytes from an explicit bit slice.
    ///
    /// This is used to prove that our matrix view is reversible.
    pub fn bytes_from_bits(bits: &[u8]) -> Option<Vec<u8>> {
        if bits.len() % 8 != 0 {
            return None;
        }

        if bits.iter().any(|bit| *bit > 1) {
            return None;
        }

        let mut output = Vec::with_capacity(bits.len() / 8);

        for chunk in bits.chunks_exact(8) {
            let mut byte = 0u8;

            for bit in chunk {
                byte = (byte << 1) | *bit;
            }

            output.push(byte);
        }

        Some(output)
    }

    pub fn full_bit_view(&self) -> Vec<u8> {
        self.bits(0, self.bit_len())
            .expect("full matrix bit range must be valid")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topology_is_1331() {
        let mut layers = [0usize; 4];

        for state in X1331State::ALL {
            layers[state.layer() as usize] += 1;
        }

        assert_eq!(layers, [1, 3, 3, 1]);
    }

    #[test]
    fn every_state_has_three_distance_one_neighbors() {
        for state in X1331State::ALL {
            let neighbors = state.neighbors();

            assert_eq!(neighbors.len(), 3);

            for neighbor in neighbors {
                assert_eq!(state.hamming_distance(neighbor), 1);
            }

            assert_ne!(neighbors[0], neighbors[1]);
            assert_ne!(neighbors[0], neighbors[2]);
            assert_ne!(neighbors[1], neighbors[2]);
        }
    }

    #[test]
    fn complements_are_exact() {
        assert_eq!(X1331State::S000.complement(), X1331State::S111);
        assert_eq!(X1331State::S001.complement(), X1331State::S110);
        assert_eq!(X1331State::S010.complement(), X1331State::S101);
        assert_eq!(X1331State::S011.complement(), X1331State::S100);

        for state in X1331State::ALL {
            assert_eq!(state.complement().complement(), state);
            assert_eq!(state.hamming_distance(state.complement()), 3);
        }
    }

    #[test]
    fn bit_coordinates_are_msb_first_and_exact() {
        let matrix = X1331BitMatrix::from_bytes(vec![0b1011_0010]);

        assert_eq!(matrix.full_bit_view(), vec![1, 0, 1, 1, 0, 0, 1, 0]);

        assert_eq!(matrix.bit(0), Some(1));
        assert_eq!(matrix.bit(7), Some(0));
        assert_eq!(matrix.bit(8), None);
    }

    #[test]
    fn raw_bytes_round_trip_through_bits() {
        let original = vec![0x00, 0x01, 0x55, 0xaa, 0xff, 0x13, 0x31, 0x80];

        let matrix = X1331BitMatrix::from_bytes(original.clone());
        let bits = matrix.full_bit_view();

        let reconstructed = X1331BitMatrix::bytes_from_bits(&bits).expect("valid bit vector");

        assert_eq!(reconstructed, original);
    }

    #[test]
    fn cells_preserve_coordinates() {
        let matrix = X1331BitMatrix::from_bytes(vec![0b000_001_11]);

        let first = matrix.cell_at(0).unwrap();
        let second = matrix.cell_at(3).unwrap();

        assert_eq!(first.start_bit, 0);
        assert_eq!(first.bits, [0, 0, 0]);
        assert_eq!(first.state, X1331State::S000);

        assert_eq!(second.start_bit, 3);
        assert_eq!(second.bits, [0, 0, 1]);
        assert_eq!(second.state, X1331State::S001);
    }

    #[test]
    fn figure_transition_is_exact() {
        let transition = FigureTransition::new(X1331State::S001, X1331State::S111);

        assert_eq!(transition.xor, X1331State::S110);
        assert_eq!(transition.hamming_distance, 2);
        assert_eq!(transition.from_layer, 1);
        assert_eq!(transition.to_layer, 3);
    }

    #[test]
    fn invalid_bit_reconstruction_is_rejected() {
        assert!(X1331BitMatrix::bytes_from_bits(&[0, 1, 0]).is_none());
        assert!(X1331BitMatrix::bytes_from_bits(&[0, 1, 2, 0, 0, 0, 0, 0]).is_none());
    }
}
