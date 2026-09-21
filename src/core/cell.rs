#[derive(Debug, Clone, Copy)]
pub struct PossibilityRegion {
    pub prefix: u64,
    pub prefix_bits: u8,
    pub total_bits: u8,
}

impl PossibilityRegion {
    pub fn root(total_bits: u8) -> Self {
        assert!(total_bits > 0 && total_bits <= 63);

        Self {
            prefix: 0,
            prefix_bits: 0,
            total_bits,
        }
    }

    pub fn remaining_bits(&self) -> u8 {
        self.total_bits - self.prefix_bits
    }

    pub fn possibilities(&self) -> u64 {
        1u64 << self.remaining_bits()
    }

    pub fn start(&self) -> u64 {
        self.prefix << self.remaining_bits()
    }

    pub fn end(&self) -> u64 {
        self.start() + self.possibilities() - 1
    }

    pub fn can_expand(&self) -> bool {
        self.remaining_bits() >= 3
    }

    pub fn expand_1331(&self) -> Vec<X1331State> {
        assert!(
            self.can_expand(),
            "Region needs at least 3 remaining bits"
        );

        (0..8)
            .map(|value| {
                let bits = [
                    (value >> 2) & 1,
                    (value >> 1) & 1,
                    value & 1,
                ];

                let weight = bits.iter().sum();

                let new_prefix = (self.prefix << 3) | value as u64;

                let region = PossibilityRegion {
                    prefix: new_prefix,
                    prefix_bits: self.prefix_bits + 3,
                    total_bits: self.total_bits,
                };

                X1331State {
                    value,
                    bits,
                    weight,
                    region,
                }
            })
            .collect()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct X1331State {
    pub value: u8,
    pub bits: [u8; 3],
    pub weight: u8,
    pub region: PossibilityRegion,
}

pub struct X1331Cell {
    pub parent: PossibilityRegion,
    pub states: Vec<X1331State>,
}

impl X1331Cell {
    pub fn from_region(parent: PossibilityRegion) -> Self {
        let states = parent.expand_1331();

        Self { parent, states }
    }

    pub fn groups(&self) -> [Vec<X1331State>; 4] {
        [
            self.states.iter().copied()
                .filter(|s| s.weight == 0).collect(),

            self.states.iter().copied()
                .filter(|s| s.weight == 1).collect(),

            self.states.iter().copied()
                .filter(|s| s.weight == 2).collect(),

            self.states.iter().copied()
                .filter(|s| s.weight == 3).collect(),
        ]
    }
}
