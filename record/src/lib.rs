//! The record format of the rdna4-w4a16 developer mode.
//!
//! In the developer mode the kernel library records where the device's time goes: each span of the caller's work
//! with the name the caller gave it and its time, and each linear product with its rows, inputs, outputs, dtype and
//! path. This crate holds the format's names (`spec`), reads a record file (`read`) and adds its spans up
//! (`total`).
//!
//! ```no_run
//! let log = rdna4_w4a16_record::read(std::path::Path::new("run.dev.jsonl")).unwrap();
//! log.check().unwrap();
//! for phase in log.phases() {
//!     println!("{} {:.1} ms", phase.phase, phase.time_ms(log.run.costs));
//! }
//! ```

pub mod read;
pub mod spec;
pub mod total;

pub use read::{Log, ReadError, parse, parse_line, read};
pub use spec::{
    Costs, Counters, Dtype, Kind, Op, Path, PhaseProduct, PhaseTotal, Product, Reason, Record, Run,
    SCHEMA, Span, Summary,
};
pub use total::Totals;
