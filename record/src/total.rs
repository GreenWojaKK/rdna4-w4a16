//! Adds the spans of a record file up the way its summary does, and checks a summary against its spans.

use std::collections::{BTreeMap, HashMap};

use crate::read::Log;
use crate::spec::{Counters, Kind, Op, Path, PhaseProduct, PhaseTotal, Product, Reason};

/// What the spans of a file add up to.
#[derive(Clone, Debug, PartialEq)]
pub struct Totals {
    /// One for each phase, in the order its first span opened.
    pub phases: Vec<PhaseTotal>,
    /// The products of each phase by kind, in the order of phases and kinds.
    pub products: Vec<PhaseProduct>,
    /// The products inside spans. `outside_calls` and `unresolved` are the summary's to tell and stay zero.
    pub counters: Counters,
}

/// Times in a line are rounded to this many milliseconds.
const ROUNDING: f64 = 0.0001;

impl Log {
    /// Adds the spans up.
    pub fn totals(&self) -> Totals {
        let mut phases: Vec<PhaseTotal> = Vec::new();
        let mut index: HashMap<&str, usize> = HashMap::new();
        // Keyed by the phase's place, as the summary orders its products.
        let mut products: BTreeMap<(usize, Kind), Product> = BTreeMap::new();
        let mut counters = Counters::default();
        for span in &self.spans {
            let place = *index.entry(span.phase.as_str()).or_insert_with(|| {
                phases.push(PhaseTotal::new(span.phase.as_str()));
                phases.len() - 1
            });
            let total = &mut phases[place];
            total.spans += 1;
            total.rows += span.rows;
            total.tokens += span.tokens;
            let mut int4_ms = 0.0;
            let mut head_ms = 0.0;
            for product in &span.products {
                match product.kind.op {
                    Op::Head => {
                        head_ms += product.ms;
                        counters.head_calls += product.calls;
                    }
                    Op::Int4 => {
                        int4_ms += product.ms;
                        if span.timed {
                            total.timed_int4_calls += product.timed_calls;
                        }
                        count_int4(&mut counters, product);
                    }
                }
                products
                    .entry((place, product.kind))
                    .and_modify(|sum| add(sum, product))
                    .or_insert(*product);
            }
            if span.timed {
                total.timed_spans += 1;
                total.timed_gpu_ms += span.gpu_ms;
                total.timed_int4_ms += int4_ms;
                total.timed_head_ms += head_ms;
            } else {
                total.wall_ms += span.wall_ms;
                total.gpu_ms += span.gpu_ms;
                total.head_ms += head_ms;
            }
        }
        let products = products
            .into_iter()
            .map(|((place, _), product)| PhaseProduct {
                phase: phases[place].phase.clone(),
                product,
            })
            .collect();
        Totals {
            phases,
            products,
            counters,
        }
    }

    /// The totals of the file: its summary's when the run ended with one, its spans' otherwise.
    pub fn phases(&self) -> Vec<PhaseTotal> {
        match &self.summary {
            Some(summary) => summary.phases.clone(),
            None => self.totals().phases,
        }
    }

    /// The products of the file by phase and kind: its summary's when the run ended with one, its spans'
    /// otherwise.
    pub fn products(&self) -> Vec<PhaseProduct> {
        match &self.summary {
            Some(summary) => summary.products.clone(),
            None => self.totals().products,
        }
    }

    /// Checks that the summary says what the spans add up to. A file without a summary has nothing to check.
    pub fn check(&self) -> Result<(), String> {
        let Some(summary) = &self.summary else {
            return Ok(());
        };
        let totals = self.totals();
        let names = |phases: &[PhaseTotal]| {
            phases
                .iter()
                .map(|total| total.phase.clone())
                .collect::<Vec<_>>()
        };
        if names(&summary.phases) != names(&totals.phases) {
            return Err(format!(
                "the summary lists the phases {:?} and the spans open {:?}",
                names(&summary.phases),
                names(&totals.phases)
            ));
        }
        for (said, added) in summary.phases.iter().zip(&totals.phases) {
            let name = said.phase.as_str();
            let spans = added.spans as f64;
            same_count(name, "spans", said.spans, added.spans)?;
            same_count(name, "rows", said.rows, added.rows)?;
            same_count(name, "tokens", said.tokens, added.tokens)?;
            same_count(name, "timed_spans", said.timed_spans, added.timed_spans)?;
            same_count(
                name,
                "timed_int4_calls",
                said.timed_int4_calls,
                added.timed_int4_calls,
            )?;
            same_time(name, "wall_ms", said.wall_ms, added.wall_ms, spans)?;
            same_time(name, "gpu_ms", said.gpu_ms, added.gpu_ms, spans)?;
            same_time(name, "head_ms", said.head_ms, added.head_ms, spans)?;
            same_time(
                name,
                "timed_gpu_ms",
                said.timed_gpu_ms,
                added.timed_gpu_ms,
                spans,
            )?;
            // A span's line holds one rounded time for each kind of product.
            let terms = self
                .spans
                .iter()
                .map(|span| span.products.len())
                .sum::<usize>() as f64;
            same_time(
                name,
                "timed_int4_ms",
                said.timed_int4_ms,
                added.timed_int4_ms,
                terms,
            )?;
            same_time(
                name,
                "timed_head_ms",
                said.timed_head_ms,
                added.timed_head_ms,
                terms,
            )?;
        }
        let said: BTreeMap<(&str, Kind), Product> = summary
            .products
            .iter()
            .map(|entry| ((entry.phase.as_str(), entry.product.kind), entry.product))
            .collect();
        if said.len() != summary.products.len() {
            return Err("the summary holds a kind of product twice".into());
        }
        if said.len() != totals.products.len() {
            return Err(format!(
                "the summary holds {} kinds of products and the spans {}",
                said.len(),
                totals.products.len()
            ));
        }
        for entry in &totals.products {
            let kind = entry.product.kind;
            let name = format!(
                "{} {} {} {}x{}x{}",
                entry.phase,
                kind.op.name(),
                kind.path.name(),
                kind.m,
                kind.k,
                kind.n
            );
            let Some(product) = said.get(&(entry.phase.as_str(), kind)) else {
                return Err(format!("{name}: in the spans and not in the summary"));
            };
            let added = &entry.product;
            same_count(&name, "calls", product.calls, added.calls)?;
            same_count(&name, "timed_calls", product.timed_calls, added.timed_calls)?;
            same_time(&name, "ms", product.ms, added.ms, self.spans.len() as f64)?;
            same_time(&name, "min_ms", product.min_ms, added.min_ms, 1.0)?;
            same_time(&name, "max_ms", product.max_ms, added.max_ms, 1.0)?;
        }
        // The summary's counters also hold the products outside every span.
        let (said, added) = (&summary.counters, &totals.counters);
        let inside = added.kernel_calls + added.head_calls + added.fallback_calls;
        let all = said.kernel_calls + said.head_calls + said.fallback_calls;
        if all != inside + said.outside_calls {
            return Err(format!(
                "the summary counts {all} products, {} of them outside spans, and the spans hold {inside}",
                said.outside_calls
            ));
        }
        let reasons = said.fallback_rows
            + said.fallback_dtype
            + said.fallback_bias
            + said.fallback_shape
            + said.fallback_launch;
        if reasons != said.fallback_calls {
            return Err(format!(
                "{} fallbacks and {reasons} reasons",
                said.fallback_calls
            ));
        }
        Ok(())
    }
}

fn count_int4(counters: &mut Counters, product: &Product) {
    if product.kind.path != Path::Unpack {
        counters.kernel_calls += product.calls;
        return;
    }
    counters.fallback_calls += product.calls;
    match product.kind.reason {
        Some(Reason::Rows) => counters.fallback_rows += product.calls,
        Some(Reason::Dtype) => counters.fallback_dtype += product.calls,
        Some(Reason::Bias) => counters.fallback_bias += product.calls,
        Some(Reason::Shape) => counters.fallback_shape += product.calls,
        Some(Reason::Launch) => counters.fallback_launch += product.calls,
        None => {}
    }
}

/// Adds the products of one kind in another span to their sum.
fn add(sum: &mut Product, more: &Product) {
    if more.timed_calls > 0 {
        sum.min_ms = if sum.timed_calls > 0 {
            sum.min_ms.min(more.min_ms)
        } else {
            more.min_ms
        };
        sum.max_ms = sum.max_ms.max(more.max_ms);
    }
    sum.calls += more.calls;
    sum.timed_calls += more.timed_calls;
    sum.ms += more.ms;
}

fn same_count(what: &str, field: &str, said: u64, added: u64) -> Result<(), String> {
    if said == added {
        return Ok(());
    }
    Err(format!(
        "{what}: the summary's {field} is {said} and the spans add up to {added}"
    ))
}

/// Compares a time of the summary with the sum of `terms` rounded times.
fn same_time(what: &str, field: &str, said: f64, added: f64, terms: f64) -> Result<(), String> {
    if (said - added).abs() <= ROUNDING * (terms + 1.0) {
        return Ok(());
    }
    Err(format!(
        "{what}: the summary's {field} is {said} and the spans add up to {added}"
    ))
}
