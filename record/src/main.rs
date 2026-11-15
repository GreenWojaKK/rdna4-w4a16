//! Prints what a record file of the developer mode holds: its phases, and the products that took the most time.
//!
//! Arguments: the record file, and optionally how many kinds of products to list, 20 unless given.

use std::path::Path;
use std::process::ExitCode;

use rdna4_w4a16_record::{Log, Op, PhaseProduct, read};

fn main() -> ExitCode {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let Some(path) = arguments.first() else {
        eprintln!("usage: rdna4-w4a16-record file.dev.jsonl [products to list]");
        return ExitCode::from(2);
    };
    let listed = match arguments.get(1).map(|text| text.parse::<usize>()) {
        None => 20,
        Some(Ok(count)) => count,
        Some(Err(_)) => {
            eprintln!("the number of products to list is not a number");
            return ExitCode::from(2);
        }
    };
    let log = match read(Path::new(path)) {
        Ok(log) => log,
        Err(error) => {
            eprintln!("{path}: {error}");
            return ExitCode::FAILURE;
        }
    };
    if let Err(reason) = log.check() {
        eprintln!("{path}: the summary does not match the spans: {reason}");
        return ExitCode::FAILURE;
    }
    print!("{}", report(&log, listed));
    ExitCode::SUCCESS
}

fn seconds(ms: Option<f64>) -> String {
    match ms {
        Some(ms) => format!("{:.2}s", ms / 1000.0),
        None => "-".into(),
    }
}

fn percent(share: Option<f64>) -> String {
    match share {
        Some(share) => format!("{:.0}%", share * 100.0),
        None => "-".into(),
    }
}

/// The text the program prints for a record file.
fn report(log: &Log, listed: usize) -> String {
    let run = &log.run;
    let mut out = format!("{} ({}), started {}\n", run.device, run.arch, run.started);
    out += &match run.every {
        0 => "no span times its int4 products\n".to_owned(),
        every => format!(
            "one span in {every} of a phase times its int4 products; a pair of events adds {:.3} ms to a product and {:.3} ms to its span\n",
            run.costs.event_ms, run.costs.pair_ms
        ),
    };
    if log.summary.is_none() {
        out += "the run ended without a summary; the totals are the spans'\n";
    }

    out += "\nphase              spans     tokens     time     head  int4 share\n";
    for phase in log.phases() {
        if phase.spans == 0 {
            continue;
        }
        out += &format!(
            "{:<16} {:>7} {:>10} {:>8} {:>8} {:>11}\n",
            phase.phase,
            phase.spans,
            phase.tokens,
            seconds(Some(phase.time_ms(run.costs))),
            seconds((phase.head_total_ms() > 0.0).then(|| phase.head_total_ms())),
            percent(phase.int4_share(run.costs)),
        );
    }

    // The products by the time all their calls take, the untimed ones taken to be like the timed ones.
    let mut products: Vec<PhaseProduct> = log.products();
    products.sort_by(|a, b| {
        let key = |entry: &PhaseProduct| entry.product.estimated_ms().unwrap_or(-1.0);
        key(b).total_cmp(&key(a))
    });
    out += "\nphase            op   path          M      K      N dtype    calls    timed  each ms     time\n";
    for entry in products.iter().take(listed) {
        let product = &entry.product;
        let kind = product.kind;
        let path = match kind.reason {
            Some(reason) => format!("{}:{}", kind.path.name(), reason.name()),
            None => kind.path.name().to_owned(),
        };
        out += &format!(
            "{:<16} {:<4} {:<11} {:>5} {:>6} {:>6} {:<5} {:>8} {:>8} {:>8} {:>8}\n",
            entry.phase,
            kind.op.name(),
            path,
            kind.m,
            kind.k,
            kind.n,
            kind.dtype.name(),
            product.calls,
            product.timed_calls,
            match product.per_call_ms() {
                Some(each) => format!("{each:.4}"),
                None => "-".into(),
            },
            seconds(product.estimated_ms()),
        );
    }
    if products.len() > listed {
        out += &format!("and {} more kinds of products\n", products.len() - listed);
    }

    let int4 = |path_is_kernel: bool| -> u64 {
        products
            .iter()
            .filter(|entry| entry.product.kind.op == Op::Int4)
            .filter(|entry| entry.product.kind.reason.is_none() == path_is_kernel)
            .map(|entry| entry.product.calls)
            .sum()
    };
    out += &format!(
        "\nint4 products in spans: {} by a kernel, {} by the checkpoint's layer\n",
        int4(true),
        int4(false)
    );
    if let Some(summary) = &log.summary {
        let counters = &summary.counters;
        out += &format!(
            "fallbacks by reason: rows {}, dtype {}, bias {}, shape {}, launch {}\n",
            counters.fallback_rows,
            counters.fallback_dtype,
            counters.fallback_bias,
            counters.fallback_shape,
            counters.fallback_launch
        );
        out += &format!(
            "products outside spans: {}; times the device did not give: {}\n",
            counters.outside_calls, counters.unresolved
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use rdna4_w4a16_record::parse;

    use super::report;

    #[test]
    fn the_report_lists_phases_and_products() {
        let log = parse(include_str!("../tests/fixtures/check.dev.jsonl")).unwrap();
        let text = report(&log, 3);
        assert!(text.contains("AMD Radeon AI PRO R9700 (gfx1201)"), "{text}");
        assert!(
            text.contains("one span in 2 of a phase times its int4 products"),
            "{text}"
        );
        for phase in ["decode", "prefill", "decision_state", "decision_suffix"] {
            assert!(
                text.lines()
                    .any(|line| line.starts_with(phase) && line.contains("s ")),
                "{text}"
            );
        }
        assert!(text.contains("unpack:rows"), "{text}");
        assert!(text.contains("more kinds of products"), "{text}");
        assert!(
            text.contains("products outside spans: 1; times the device did not give: 0"),
            "{text}"
        );
    }
}
