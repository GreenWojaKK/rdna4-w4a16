//! Reads a record file into its run, spans and summary.

use std::fmt;
use std::fs;
use std::io;

use serde_json::{Map, Value};

use std::collections::HashSet;

use crate::spec::{
    Costs, Counters, Dtype, Kind, Op, Path, PhaseProduct, PhaseTotal, Product, Reason, Record, Run,
    SCHEMA, Span, Summary,
};

/// Why a record file could not be read.
#[derive(Debug)]
pub enum ReadError {
    Io(io::Error),
    /// A line that is not a record of the format: its 1-based number and what is wrong with it.
    Line {
        line: usize,
        reason: String,
    },
    /// The lines are records but do not make a record file.
    File(String),
}

impl fmt::Display for ReadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReadError::Io(error) => write!(f, "{error}"),
            ReadError::Line { line, reason } => write!(f, "line {line}: {reason}"),
            ReadError::File(reason) => write!(f, "{reason}"),
        }
    }
}

impl std::error::Error for ReadError {}

impl From<io::Error> for ReadError {
    fn from(error: io::Error) -> ReadError {
        ReadError::Io(error)
    }
}

/// A record file: the run, its spans in order, and the summary when the run ended with one.
#[derive(Clone, Debug, PartialEq)]
pub struct Log {
    pub run: Run,
    pub spans: Vec<Span>,
    pub summary: Option<Summary>,
}

/// Reads the record file at a path.
pub fn read(path: &std::path::Path) -> Result<Log, ReadError> {
    parse(&fs::read_to_string(path)?)
}

/// Reads the text of a record file. The first line is the run, spans follow in the order of their numbers, and a
/// summary is the last line. A file whose run was cut short has no summary.
pub fn parse(text: &str) -> Result<Log, ReadError> {
    let mut run = None;
    let mut spans: Vec<Span> = Vec::new();
    let mut summary = None;
    for (index, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let at = |reason: String| ReadError::Line {
            line: index + 1,
            reason,
        };
        if summary.is_some() {
            return Err(at("a record follows the summary".into()));
        }
        match parse_line(line).map_err(at)? {
            Record::Run(_) if run.is_some() => return Err(at("a second run".into())),
            Record::Run(first) => run = Some(first),
            _ if run.is_none() => return Err(at("the file does not start with the run".into())),
            Record::Span(span) => {
                if span.seq != spans.len() as u64 {
                    return Err(at(format!(
                        "span {} where span {} belongs",
                        span.seq,
                        spans.len()
                    )));
                }
                spans.push(span);
            }
            Record::Summary(last) => summary = Some(last),
        }
    }
    let run = run.ok_or_else(|| ReadError::File("the file holds no run".into()))?;
    if let Some(summary) = &summary {
        if summary.spans != spans.len() as u64 {
            return Err(ReadError::File(format!(
                "the summary counts {} spans and the file holds {}",
                summary.spans,
                spans.len()
            )));
        }
        if summary.every != run.every || summary.costs != run.costs {
            return Err(ReadError::File(
                "the summary's costs are not the run's".into(),
            ));
        }
    }
    Ok(Log {
        run,
        spans,
        summary,
    })
}

/// Reads one line of a record file.
pub fn parse_line(line: &str) -> Result<Record, String> {
    let value: Value = serde_json::from_str(line).map_err(|error| error.to_string())?;
    let object = value.as_object().ok_or("the line is not a JSON object")?;
    match text(object, "record")? {
        "run" => run(object).map(Record::Run),
        "span" => span(object).map(Record::Span),
        "summary" => summary(object).map(Record::Summary),
        other => Err(format!("unknown record {other:?}")),
    }
}

type Fields = Map<String, Value>;

fn field<'a>(object: &'a Fields, name: &str) -> Result<&'a Value, String> {
    object.get(name).ok_or_else(|| format!("no field {name:?}"))
}

fn text<'a>(object: &'a Fields, name: &str) -> Result<&'a str, String> {
    field(object, name)?
        .as_str()
        .ok_or_else(|| format!("{name:?} is not a string"))
}

fn count(object: &Fields, name: &str) -> Result<u64, String> {
    field(object, name)?
        .as_u64()
        .ok_or_else(|| format!("{name:?} is not a count"))
}

/// A time in milliseconds: a number that is not negative.
fn time(object: &Fields, name: &str) -> Result<f64, String> {
    field(object, name)?
        .as_f64()
        .filter(|ms| ms.is_finite() && *ms >= 0.0)
        .ok_or_else(|| format!("{name:?} is not a time"))
}

fn flag(object: &Fields, name: &str) -> Result<bool, String> {
    field(object, name)?
        .as_bool()
        .ok_or_else(|| format!("{name:?} is not true or false"))
}

fn objects<'a>(object: &'a Fields, name: &str) -> Result<Vec<&'a Fields>, String> {
    field(object, name)?
        .as_array()
        .ok_or_else(|| format!("{name:?} is not a list"))?
        .iter()
        .map(|item| {
            item.as_object()
                .ok_or_else(|| format!("{name:?} holds something other than objects"))
        })
        .collect()
}

/// The name of a phase: a string that is not empty.
fn phase_name(object: &Fields) -> Result<String, String> {
    let name = text(object, "phase")?;
    if name.is_empty() {
        return Err("\"phase\" is empty".into());
    }
    Ok(name.to_owned())
}

/// A name of the format, such as a path's.
fn named<T>(
    object: &Fields,
    name: &str,
    from_name: impl Fn(&str) -> Option<T>,
) -> Result<T, String> {
    let value = text(object, name)?;
    from_name(value).ok_or_else(|| format!("unknown {name} {value:?}"))
}

fn run(object: &Fields) -> Result<Run, String> {
    let schema = count(object, "schema")?;
    if schema != SCHEMA {
        return Err(format!(
            "the file is of format {schema} and this reader of format {SCHEMA}"
        ));
    }
    Ok(Run {
        started: text(object, "started")?.to_owned(),
        device: text(object, "device")?.to_owned(),
        arch: text(object, "arch")?.to_owned(),
        every: count(object, "every")?,
        costs: costs(object)?,
    })
}

fn costs(object: &Fields) -> Result<Costs, String> {
    Ok(Costs {
        event_ms: time(object, "event_ms")?,
        pair_ms: time(object, "pair_ms")?,
    })
}

fn product(object: &Fields) -> Result<Product, String> {
    let op = named(object, "op", Op::from_name)?;
    let path = named(object, "path", Path::from_name)?;
    let reason = match object.get("reason") {
        Some(_) => Some(named(object, "reason", Reason::from_name)?),
        None => None,
    };
    // The head's product is the dense one, and only the checkpoint's own path has a reason.
    if (op == Op::Head) != (path == Path::Dense) {
        return Err(format!(
            "a {} product on the path {}",
            op.name(),
            path.name()
        ));
    }
    if reason.is_some() != (path == Path::Unpack) {
        return Err(format!(
            "the path {} {} a reason",
            path.name(),
            if reason.is_some() { "with" } else { "without" }
        ));
    }
    let product = Product {
        kind: Kind {
            op,
            path,
            reason,
            m: count(object, "m")?,
            k: count(object, "k")?,
            n: count(object, "n")?,
            dtype: named(object, "dtype", Dtype::from_name)?,
        },
        calls: count(object, "calls")?,
        timed_calls: count(object, "timed_calls")?,
        ms: time(object, "ms")?,
        min_ms: time(object, "min_ms")?,
        max_ms: time(object, "max_ms")?,
    };
    if product.timed_calls > product.calls {
        return Err(format!(
            "{} timed calls of {}",
            product.timed_calls, product.calls
        ));
    }
    Ok(product)
}

fn span(object: &Fields) -> Result<Span, String> {
    Ok(Span {
        seq: count(object, "seq")?,
        phase: phase_name(object)?,
        rows: count(object, "rows")?,
        tokens: count(object, "tokens")?,
        wall_ms: time(object, "wall_ms")?,
        gpu_ms: time(object, "gpu_ms")?,
        timed: flag(object, "timed")?,
        products: objects(object, "products")?
            .into_iter()
            .map(product)
            .collect::<Result<_, _>>()?,
    })
}

fn phase_total(object: &Fields) -> Result<PhaseTotal, String> {
    let total = PhaseTotal {
        phase: phase_name(object)?,
        spans: count(object, "spans")?,
        rows: count(object, "rows")?,
        tokens: count(object, "tokens")?,
        wall_ms: time(object, "wall_ms")?,
        gpu_ms: time(object, "gpu_ms")?,
        head_ms: time(object, "head_ms")?,
        timed_spans: count(object, "timed_spans")?,
        timed_gpu_ms: time(object, "timed_gpu_ms")?,
        timed_int4_ms: time(object, "timed_int4_ms")?,
        timed_head_ms: time(object, "timed_head_ms")?,
        timed_int4_calls: count(object, "timed_int4_calls")?,
    };
    if total.timed_spans > total.spans {
        return Err(format!(
            "{}: {} timed spans of {}",
            total.phase, total.timed_spans, total.spans
        ));
    }
    Ok(total)
}

fn counters(object: &Fields) -> Result<Counters, String> {
    Ok(Counters {
        kernel_calls: count(object, "kernel_calls")?,
        head_calls: count(object, "head_calls")?,
        fallback_calls: count(object, "fallback_calls")?,
        fallback_rows: count(object, "fallback_rows")?,
        fallback_dtype: count(object, "fallback_dtype")?,
        fallback_bias: count(object, "fallback_bias")?,
        fallback_shape: count(object, "fallback_shape")?,
        fallback_launch: count(object, "fallback_launch")?,
        outside_calls: count(object, "outside_calls")?,
        unresolved: count(object, "unresolved")?,
    })
}

fn summary(object: &Fields) -> Result<Summary, String> {
    let phases: Vec<PhaseTotal> = objects(object, "phases")?
        .into_iter()
        .map(phase_total)
        .collect::<Result<_, _>>()?;
    let mut seen = HashSet::new();
    for total in &phases {
        if !seen.insert(total.phase.as_str()) {
            return Err(format!(
                "the summary lists the phase {:?} twice",
                total.phase
            ));
        }
    }
    let products = objects(object, "products")?
        .into_iter()
        .map(|item| {
            Ok(PhaseProduct {
                phase: phase_name(item)?,
                product: product(item)?,
            })
        })
        .collect::<Result<_, String>>()?;
    let counters = counters(
        field(object, "counters")?
            .as_object()
            .ok_or("\"counters\" is not an object")?,
    )?;
    Ok(Summary {
        elapsed_ms: time(object, "elapsed_ms")?,
        spans: count(object, "spans")?,
        every: count(object, "every")?,
        costs: costs(object)?,
        phases,
        products,
        counters,
    })
}
