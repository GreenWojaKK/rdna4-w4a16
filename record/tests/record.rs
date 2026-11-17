//! Reads a record file the kernel library wrote on the device, and lines that break the
//! format.

use rdna4_w4a16_record::{Dtype, Op, Path, ReadError, Reason, Record, parse, parse_line};

const RECORDED: &str = include_str!("fixtures/check.dev.jsonl");

/// The record the device check wrote: two spans in each of four phases, the second timing its int4 products, each
/// over 40 rounds of three layers; the decode and prefill spans end with the head.
#[test]
fn a_recorded_file_reads_and_its_summary_matches_its_spans() {
    let log = parse(RECORDED).unwrap();
    assert_eq!(log.run.arch, "gfx1201");
    assert_eq!(log.run.every, 2);
    assert!(log.run.costs.event_ms > 0.0 && log.run.costs.pair_ms > log.run.costs.event_ms);
    assert_eq!(log.spans.len(), 8);
    assert_eq!(
        log.spans.iter().map(|span| span.timed).collect::<Vec<_>>(),
        [false, true].repeat(4)
    );
    log.check().unwrap();

    let summary = log.summary.as_ref().unwrap();
    // The summary's times are sums of what the lines round.
    let names: Vec<&str> = summary.phases.iter().map(|p| p.phase.as_str()).collect();
    assert_eq!(
        names,
        ["decode", "prefill", "decision_suffix", "decision_state"]
    );
    for added in log.totals().phases {
        let said = summary
            .phases
            .iter()
            .find(|p| p.phase == added.phase)
            .unwrap();
        assert!((said.gpu_ms - added.gpu_ms).abs() < 0.001);
        assert!((said.timed_int4_ms - added.timed_int4_ms).abs() < 0.001);
    }
    for phase in &summary.phases {
        assert_eq!(
            (phase.spans, phase.timed_spans),
            (2, 1),
            "{:?}",
            phase.phase
        );
        assert_eq!(phase.timed_int4_calls, 120, "{:?}", phase.phase);
        // The time of a phase is that of its spans, the timed one without its events.
        let time = phase.time_ms(log.run.costs);
        assert!(time > phase.gpu_ms && time < phase.gpu_ms + phase.timed_gpu_ms);
        let share = phase.int4_share(log.run.costs).unwrap();
        assert!(share > 0.3 && share <= 1.0, "{:?}: {share}", phase.phase);
    }
    assert_eq!(summary.counters.kernel_calls, 3 * 3 * 80 + 1);
    assert_eq!(summary.counters.fallback_calls, 3 * 80);
    assert_eq!(summary.counters.fallback_rows, 3 * 80);
    assert_eq!(summary.counters.head_calls, 4);
    assert_eq!(summary.counters.outside_calls, 1);
}

#[test]
fn products_carry_their_rows_sizes_path_and_reason() {
    let log = parse(RECORDED).unwrap();
    let products = log.products();
    let find = |phase: &str, path: Path, n: u64| {
        products
            .iter()
            .find(|entry| {
                entry.phase == phase && entry.product.kind.path == path && entry.product.kind.n == n
            })
            .unwrap_or_else(|| panic!("no {path:?} product of {n} outputs in {phase}"))
            .product
    };

    let dot = find("decode", Path::Dot, 17408);
    assert_eq!(
        (dot.kind.m, dot.kind.k, dot.kind.dtype),
        (1, 5120, Dtype::Bf16)
    );
    assert_eq!((dot.calls, dot.timed_calls), (80, 40));
    assert!(dot.per_call_ms().unwrap() > 0.0);
    assert!((dot.estimated_ms().unwrap() - 2.0 * dot.ms).abs() < 1e-9);

    let tile = find("prefill", Path::Tile2, 17408);
    assert_eq!(
        (tile.kind.m, tile.kind.op, tile.kind.reason),
        (250, Op::Int4, None)
    );

    let unpacked = find("decision_state", Path::Unpack, 5120);
    assert_eq!((unpacked.kind.m, unpacked.kind.k), (1500, 17408));
    assert_eq!(unpacked.kind.reason, Some(Reason::Rows));

    let head = find("decode", Path::Dense, 248320);
    assert_eq!(
        (head.kind.op, head.kind.m, head.kind.k),
        (Op::Head, 1, 5120)
    );
    // The head is timed in every span.
    assert_eq!((head.calls, head.timed_calls), (2, 2));
}

#[test]
fn a_file_cut_short_has_its_spans_totals() {
    let lines: Vec<&str> = RECORDED.lines().collect();
    let cut = lines[..lines.len() - 1].join("\n");
    let log = parse(&cut).unwrap();
    assert!(log.summary.is_none());
    log.check().unwrap();
    let whole = parse(RECORDED).unwrap();
    assert_eq!(log.products().len(), whole.products().len());
    let decode = log
        .phases()
        .into_iter()
        .find(|p| p.phase == "decode")
        .unwrap();
    assert_eq!(decode.spans, 2);
}

#[test]
fn a_summary_that_differs_from_its_spans_fails_the_check() {
    let lines: Vec<&str> = RECORDED.lines().collect();
    // Without the last span the summary counts one span too many.
    let mut short = lines.clone();
    short.remove(lines.len() - 2);
    assert!(
        matches!(parse(&short.join("\n")), Err(ReadError::File(reason)) if reason.contains("spans"))
    );

    // A product's calls changed in a span.
    let changed = RECORDED.replacen("\"calls\":40,", "\"calls\":41,", 1);
    let reason = parse(&changed).unwrap().check().unwrap_err();
    assert!(reason.contains("calls"), "{reason}");

    // The summary lists a phase twice.
    let mut twice = lines.clone();
    let last = twice
        .pop()
        .unwrap()
        .replacen("\"phase\":\"prefill\"", "\"phase\":\"decode\"", 1);
    twice.push(&last);
    assert!(
        matches!(parse(&twice.join("\n")), Err(ReadError::Line { reason, .. }) if reason.contains("twice"))
    );
}

#[test]
fn lines_out_of_order_are_refused() {
    let lines: Vec<&str> = RECORDED.lines().collect();
    let error = |text: String| match parse(&text) {
        Err(ReadError::Line { line, reason }) => (line, reason),
        other => panic!("{other:?}"),
    };
    assert_eq!(error(lines[1..].join("\n")).0, 1);
    assert!(
        error(format!("{}\n{}", lines[0], lines[0]))
            .1
            .contains("second run")
    );
    assert!(
        error(format!("{}\n{}", lines[0], lines[2]))
            .1
            .contains("span 1 where span 0 belongs")
    );
    assert!(
        error(format!("{RECORDED}{}\n", lines[1]))
            .1
            .contains("follows the summary")
    );
    assert!(matches!(parse(""), Err(ReadError::File(_))));
}

#[test]
fn a_line_names_what_is_wrong_with_it() {
    let run = r#"{"record":"run","schema":2,"started":"2026-11-14T05:00:00Z","device":"d","arch":"gfx1201","every":8,"event_ms":0.2,"pair_ms":0.4}"#;
    assert!(matches!(parse_line(run), Ok(Record::Run(_))));
    let bad = |line: &str| parse_line(line).unwrap_err();
    assert!(bad(&run.replace("\"schema\":2", "\"schema\":3")).contains("format 3"));
    assert!(bad(&run.replace("\"every\":8,", "")).contains("no field \"every\""));
    assert!(bad(&run.replace("\"event_ms\":0.2", "\"event_ms\":-1")).contains("not a time"));
    assert!(bad("[]").contains("not a JSON object"));
    assert!(bad(r#"{"record":"step"}"#).contains("unknown record"));

    let span = r#"{"record":"span","seq":0,"phase":"decode","rows":16,"tokens":16,"wall_ms":100.5,"gpu_ms":100.1,"timed":true,"products":[{"op":"int4","path":"tile1","m":16,"k":5120,"n":17408,"dtype":"bf16","calls":128,"timed_calls":128,"ms":18.3,"min_ms":0.1,"max_ms":0.3}]}"#;
    let Ok(Record::Span(read)) = parse_line(span) else {
        panic!("the span does not read");
    };
    assert_eq!(
        (read.phase.as_str(), read.rows, read.products[0].kind.path),
        ("decode", 16, Path::Tile1)
    );
    assert!(bad(&span.replace("\"decode\"", "\"\"")).contains("\"phase\" is empty"));
    assert!(bad(&span.replace("tile1", "tile9")).contains("unknown path"));
    assert!(bad(&span.replace("\"bf16\"", "\"int8\"")).contains("unknown dtype"));
    assert!(
        bad(&span.replace("\"timed_calls\":128", "\"timed_calls\":129"))
            .contains("129 timed calls of 128")
    );
    // Only the checkpoint's own path has a reason, and only the head is dense.
    assert!(
        bad(&span.replace("\"tile1\"", "\"tile1\",\"reason\":\"rows\"")).contains("with a reason")
    );
    assert!(bad(&span.replace("\"tile1\"", "\"unpack\"")).contains("without a reason"));
    assert!(
        bad(&span.replace("\"tile1\"", "\"dense\"")).contains("int4 product on the path dense")
    );
}
