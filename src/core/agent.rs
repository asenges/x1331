use crate::core::cell::X1331Cell;

#[derive(Debug, Clone)]
pub struct Proposal {
    pub agent: &'static str,
    pub method: &'static str,
    pub parameters: &'static str,
    pub state_value: u8,
    pub score: f64,
}

impl Proposal {
    pub fn method_key(&self) -> String {
        format!(
            "{}::{}::{}",
            self.agent,
            self.method,
            self.parameters
        )
    }
}

pub trait Agent {
    fn name(&self) -> &'static str;
    fn propose(&self, cell: &X1331Cell) -> Vec<Proposal>;
}

// Baseline. It contains no information by design.
pub struct UniformAgent;

impl Agent for UniformAgent {
    fn name(&self) -> &'static str {
        "control-agent"
    }

    fn propose(&self, cell: &X1331Cell) -> Vec<Proposal> {
        cell.states
            .iter()
            .map(|state| Proposal {
                agent: self.name(),
                method: "uniform",
                parameters: "none",
                state_value: state.value,
                score: 1.0 / 8.0,
            })
            .collect()
    }
}

// One agent can execute several methods.
pub struct BinaryAgent;

impl Agent for BinaryAgent {
    fn name(&self) -> &'static str {
        "binary-agent"
    }

    fn propose(&self, cell: &X1331Cell) -> Vec<Proposal> {
        let mut proposals = Vec::new();

        // METHOD 1: Hamming balance.
        for state in &cell.states {
            let distance =
                (state.weight as f64 - 1.5).abs();

            proposals.push(Proposal {
                agent: self.name(),
                method: "hamming-balance",
                parameters: "center=1.5",
                state_value: state.value,
                score: 1.0 / (1.0 + distance),
            });
        }

        // METHOD 2: bit transitions.
        // 010 / 101 have two transitions.
        // 001 / 011 / 100 / 110 have one.
        // 000 / 111 have zero.
        for state in &cell.states {
            let b = state.bits;

            let transitions =
                (b[0] != b[1]) as u8 +
                (b[1] != b[2]) as u8;

            proposals.push(Proposal {
                agent: self.name(),
                method: "bit-transitions",
                parameters: "window=3",
                state_value: state.value,
                score: (transitions as f64 + 1.0) / 3.0,
            });
        }

        proposals
    }
}
