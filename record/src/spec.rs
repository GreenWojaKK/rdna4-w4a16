//! The names and the shapes of the developer mode's records.
//!
//! The kernel library writes the record file (`kernels/instrument.cpp`), one JSON object per line: the run, then a
//! line for each span, then the summary. A span is one piece of the caller's work under a name the caller gives; the
//! spans of one name form a phase. A product is a linear product on the device: M rows of input by K inputs by N
//! outputs.

/// The version of the record format this crate reads.
pub const SCHEMA: u64 = 2;

/// What kind of product a product is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Op {
    /// A layer of int4 weights.
    Int4,
    /// A product on BF16 weights that the caller runs and times itself, such as a model's vocabulary head.
    Head,
}

impl Op {
    pub const ALL: [Op; 2] = [Op::Int4, Op::Head];

    pub fn name(self) -> &'static str {
        match self {
            Op::Int4 => "int4",
            Op::Head => "head",
        }
    }

    pub fn from_name(name: &str) -> Option<Op> {
        Op::ALL.into_iter().find(|op| op.name() == name)
    }
}

/// The path a product took.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Path {
    /// The checkpoint's own layer: its kernel unpacks the whole weight to BF16 and the product follows.
    Unpack,
    /// The kernel for one row of input, on the BF16 dot product instruction.
    Dot,
    /// The matrix kernel with one weight tile to a workgroup.
    Tile1,
    /// The matrix kernel with two weight tiles to a workgroup.
    Tile2,
    /// A product on BF16 weights, which is the head's.
    Dense,
}

impl Path {
    pub const ALL: [Path; 5] = [
        Path::Unpack,
        Path::Dot,
        Path::Tile1,
        Path::Tile2,
        Path::Dense,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Path::Unpack => "unpack",
            Path::Dot => "dot",
            Path::Tile1 => "tile1",
            Path::Tile2 => "tile2",
            Path::Dense => "dense",
        }
    }

    pub fn from_name(name: &str) -> Option<Path> {
        Path::ALL.into_iter().find(|path| path.name() == name)
    }
}

/// Why an int4 product took the checkpoint's own layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Reason {
    /// More rows of input than a kernel takes.
    Rows,
    /// The input is not BF16.
    Dtype,
    /// The layer adds a bias.
    Bias,
    /// The layer's sizes or its groups of columns are not ones a kernel takes.
    Shape,
    /// A kernel's launch failed.
    Launch,
}

impl Reason {
    pub const ALL: [Reason; 5] = [
        Reason::Rows,
        Reason::Dtype,
        Reason::Bias,
        Reason::Shape,
        Reason::Launch,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Reason::Rows => "rows",
            Reason::Dtype => "dtype",
            Reason::Bias => "bias",
            Reason::Shape => "shape",
            Reason::Launch => "launch",
        }
    }

    pub fn from_name(name: &str) -> Option<Reason> {
        Reason::ALL.into_iter().find(|reason| reason.name() == name)
    }
}

/// The dtype of a product's input.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Dtype {
    Other,
    Bf16,
    F16,
    F32,
}

impl Dtype {
    pub const ALL: [Dtype; 4] = [Dtype::Other, Dtype::Bf16, Dtype::F16, Dtype::F32];

    pub fn name(self) -> &'static str {
        match self {
            Dtype::Other => "other",
            Dtype::Bf16 => "bf16",
            Dtype::F16 => "f16",
            Dtype::F32 => "f32",
        }
    }

    pub fn from_name(name: &str) -> Option<Dtype> {
        Dtype::ALL.into_iter().find(|dtype| dtype.name() == name)
    }
}

/// The first line: the run the file records.
#[derive(Clone, Debug, PartialEq)]
pub struct Run {
    /// When recording started, in UTC.
    pub started: String,
    pub device: String,
    /// The device's architecture, such as `gfx1201`.
    pub arch: String,
    /// One span in every so many of a phase has its int4 products timed. None when zero.
    pub every: u64,
    pub costs: Costs,
}

/// What the events that time a product cost, measured at the start of the run on launches that compute nothing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Costs {
    /// What a pair of events adds to the time of what lies between them, in milliseconds. It is taken off the time
    /// of each timed product, so a product's time is exact to about this much.
    pub event_ms: f64,
    /// What a timed product adds to the time of its span, in milliseconds. A timed span's time holds this once
    /// for each of its timed int4 products.
    pub pair_ms: f64,
}

/// What a product was: its kind, its path, and M rows of input by K inputs by N outputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Kind {
    pub op: Op,
    pub path: Path,
    /// Set when an int4 product took the checkpoint's own layer.
    pub reason: Option<Reason>,
    pub m: u64,
    pub k: u64,
    pub n: u64,
    pub dtype: Dtype,
}

/// The products of one kind in a span, or in a phase of the summary.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Product {
    pub kind: Kind,
    /// How many ran.
    pub calls: u64,
    /// How many of them were timed. The head is always timed; an int4 product is timed in the timed spans.
    pub timed_calls: u64,
    /// The device's time for the timed ones, in milliseconds, with the events' cost taken off.
    pub ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
}

impl Product {
    /// The time of one product, from the timed ones.
    pub fn per_call_ms(&self) -> Option<f64> {
        (self.timed_calls > 0).then(|| self.ms / self.timed_calls as f64)
    }

    /// The time of all of them, taking the others to be like the timed ones.
    pub fn estimated_ms(&self) -> Option<f64> {
        self.per_call_ms().map(|each| each * self.calls as f64)
    }
}

/// One piece of the caller's work.
#[derive(Clone, Debug, PartialEq)]
pub struct Span {
    /// Its place in the file, from zero.
    pub seq: u64,
    /// The name the caller gave it: the phase it belongs to.
    pub phase: String,
    /// Rows of the batch the caller states, such as the sequences of a decode step.
    pub rows: u64,
    /// Tokens the span computes, over all its rows and with their padding.
    pub tokens: u64,
    /// The host's time from the span's start until the device reached its end, in milliseconds.
    pub wall_ms: f64,
    /// The device's time between the span's first and last event, in milliseconds.
    pub gpu_ms: f64,
    /// Whether its int4 products were timed. Such a span's own time holds the cost of their events.
    pub timed: bool,
    pub products: Vec<Product>,
}

/// The spans of a phase, added up.
#[derive(Clone, Debug, PartialEq)]
pub struct PhaseTotal {
    pub phase: String,
    /// Every span, timed or not.
    pub spans: u64,
    pub rows: u64,
    pub tokens: u64,
    /// The time of the spans whose int4 products were not timed.
    pub wall_ms: f64,
    pub gpu_ms: f64,
    /// The head's time in those spans.
    pub head_ms: f64,
    /// The spans whose int4 products were timed, and their time, which holds the events' cost.
    pub timed_spans: u64,
    pub timed_gpu_ms: f64,
    /// What the int4 products and the head took in the timed spans, and how many int4 products were timed.
    pub timed_int4_ms: f64,
    pub timed_head_ms: f64,
    pub timed_int4_calls: u64,
}

impl PhaseTotal {
    pub fn new(phase: impl Into<String>) -> PhaseTotal {
        PhaseTotal {
            phase: phase.into(),
            spans: 0,
            rows: 0,
            tokens: 0,
            wall_ms: 0.0,
            gpu_ms: 0.0,
            head_ms: 0.0,
            timed_spans: 0,
            timed_gpu_ms: 0.0,
            timed_int4_ms: 0.0,
            timed_head_ms: 0.0,
            timed_int4_calls: 0,
        }
    }

    /// The spans whose time is free of the events' cost.
    pub fn whole_spans(&self) -> u64 {
        self.spans - self.timed_spans
    }

    /// The time of the timed spans with the cost of their events taken off.
    pub fn timed_ms(&self, costs: Costs) -> f64 {
        (self.timed_gpu_ms - costs.pair_ms * self.timed_int4_calls as f64).max(0.0)
    }

    /// The device's time for the phase: its whole spans, and its timed spans without their events' cost.
    pub fn time_ms(&self, costs: Costs) -> f64 {
        self.gpu_ms + self.timed_ms(costs)
    }

    /// The head's time for the phase. The head is timed in every span.
    pub fn head_total_ms(&self) -> f64 {
        self.head_ms + self.timed_head_ms
    }

    /// The share of the phase's time its int4 products take, from the timed spans: what the products took against
    /// what those spans took without their events' cost.
    pub fn int4_share(&self, costs: Costs) -> Option<f64> {
        let timed = self.timed_ms(costs);
        (self.timed_int4_calls > 0 && timed > 0.0).then(|| (self.timed_int4_ms / timed).min(1.0))
    }
}

/// The products of one kind in one phase, added up.
#[derive(Clone, Debug, PartialEq)]
pub struct PhaseProduct {
    pub phase: String,
    pub product: Product,
}

/// Counts over the whole run.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counters {
    /// Int4 products a kernel ran.
    pub kernel_calls: u64,
    pub head_calls: u64,
    /// Int4 products the checkpoint's own layer ran, and how many for each reason.
    pub fallback_calls: u64,
    pub fallback_rows: u64,
    pub fallback_dtype: u64,
    pub fallback_bias: u64,
    pub fallback_shape: u64,
    pub fallback_launch: u64,
    /// Products outside every span. They are in no span's line.
    pub outside_calls: u64,
    /// Times the device did not give.
    pub unresolved: u64,
}

/// The last line: the run added up.
#[derive(Clone, Debug, PartialEq)]
pub struct Summary {
    /// The host's time from the start of recording to its end, in milliseconds.
    pub elapsed_ms: f64,
    pub spans: u64,
    /// The run's `every` and costs again, so the summary reads on its own.
    pub every: u64,
    pub costs: Costs,
    /// One for each phase, in the order its first span opened.
    pub phases: Vec<PhaseTotal>,
    pub products: Vec<PhaseProduct>,
    pub counters: Counters,
}

/// One line of the record file.
#[derive(Clone, Debug, PartialEq)]
pub enum Record {
    Run(Run),
    Span(Span),
    Summary(Summary),
}
