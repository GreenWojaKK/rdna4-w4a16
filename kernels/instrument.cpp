// The developer mode's profiler, statistics and export.
//
// Time is taken on the device with HIP events on the default stream, the stream the caller's products run on: the
// time between two reached events is the device's own. A span is one piece of the caller's work under a name the
// caller gives, such as a decode step; the spans of one name form a phase. Its events are read when it ends, after its
// last event has been reached, so recording waits for the device only where the caller reads results back anyway.
//
// An event makes the device finish what is queued before it, which a product pays for: a pair of events around a
// product of a tenth of a millisecond more than doubles it. So every span is timed as a whole, with its head product,
// and every int4 product is counted with what it was, but only one span in every few has its int4 products timed.
// The time such a span takes holds the events' cost and is kept apart from the others'.
//
// Two costs are measured at the start, on launches that compute nothing: what a pair of events adds to the time
// between them, which is taken off each timed product, and what a timed product adds to the time of its span, which
// readers take off the timed spans.
//
// A span and each timed product are also ROCTx ranges, which the ROCm profilers show beside the kernels they trace.
//
// Each span is one line of the record file, with its products grouped by what they were. The summary, the file's
// last line, adds the spans up by phase and the products by phase and kind. Phases are listed in the order their
// first span opened. One thread drives all of this: the caller's.

#include "instrument.h"

#include <hip/hip_runtime_api.h>

#include "api.h"

#if RDNA4_W4A16_ROCTX
#include <roctracer/roctx.h>
#else
// The ROCm SDK for Windows holds no ROCTx: spans and timed products are then not marked for profilers.
static int roctxRangePushA(const char*) { return 0; }
static int roctxRangePop() { return 0; }
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <compare>
#include <cstddef>
#include <cstdio>
#include <format>
#include <limits>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace rdna4_w4a16::dev {
namespace {

constexpr int kSchema = 2;
constexpr int kCalibrationRounds = 5;
constexpr int kCalibrationLaunches = 64;
constexpr std::array<std::string_view, 2> kOpNames = {"int4", "head"};
constexpr std::array<std::string_view, 5> kPathNames = {"unpack", "dot", "tile1", "tile2", "dense"};
constexpr std::array<std::string_view, kReasons> kReasonNames = {"", "rows", "dtype", "bias", "shape", "launch"};
constexpr std::array<std::string_view, 4> kDtypeNames = {"other", "bf16", "f16", "f32"};

using Clock = std::chrono::steady_clock;

double milliseconds(Clock::duration time) { return std::chrono::duration<double, std::milli>(time).count(); }

// What a product was, and in the totals also the phase it ran in.
struct Key {
    int phase, op, path, reason, m, k, n, dtype;
    auto operator<=>(const Key&) const = default;
};

Key key_of(int phase, const Product& product) {
    return {phase,     static_cast<int>(product.op), static_cast<int>(product.path), static_cast<int>(product.reason),
            product.m, product.k,                    product.n,                      static_cast<int>(product.dtype)};
}

// The products of one kind: how many ran, and the time of those that were timed.
struct Stat {
    long long calls = 0;
    long long timed = 0;
    double ms = 0;
    double low = std::numeric_limits<double>::infinity();
    double high = 0;

    void add(double value) {
        ++timed;
        ms += value;
        low = std::min(low, value);
        high = std::max(high, value);
    }
    void merge(const Stat& other) {
        calls += other.calls;
        timed += other.timed;
        ms += other.ms;
        low = std::min(low, other.low);
        high = std::max(high, other.high);
    }
};

// The products of one kind within the open span.
struct Group {
    Product product;
    Stat stat;
    std::string label;  // The name of its ROCTx ranges.
};

struct Pending {
    hipEvent_t start;
    hipEvent_t stop;
    std::size_t group;
};

// The spans of a phase: the spans of one name. Device and wall time are those of the spans whose int4 products were not timed; the timed
// spans' are kept apart because they hold the events' cost.
struct PhaseTotal {
    long long spans = 0;
    long long rows = 0;
    long long tokens = 0;
    double wall_ms = 0;
    double gpu_ms = 0;
    double head_ms = 0;
    long long timed_spans = 0;
    double timed_gpu_ms = 0;
    double timed_int4_ms = 0;
    double timed_head_ms = 0;
    long long timed_int4_calls = 0;
    std::string name;
};

struct State {
    std::FILE* file = nullptr;
    bool on = false;
    int every = 0;        // One span in every so many of a phase has its int4 products timed. None when zero.
    double event_ms = 0;  // What a pair of events adds to the time of what lies between them.
    double pair_ms = 0;   // What a timed product adds to the time of its span.
    Clock::time_point began;
    std::vector<hipEvent_t> spare;  // Events to record again.

    // The open span.
    bool open = false;
    bool span_timed = false;
    std::size_t phase = 0;  // Index into phases.
    long long rows = 0;
    long long tokens = 0;
    hipEvent_t start = nullptr;
    Clock::time_point wall;
    std::vector<Group> groups;
    std::vector<Pending> pending;

    // The open product.
    int depth = 0;
    bool timed = false;
    hipEvent_t product_start = nullptr;
    std::size_t product_group = 0;

    // Totals.
    long long spans = 0;
    std::vector<PhaseTotal> phases;
    std::map<Key, Stat> products;
    std::array<long long, kReasons> fallbacks{};
    long long kernel_calls = 0;
    long long head_calls = 0;
    long long outside_calls = 0;  // Products outside every span: counted only here.
    long long unresolved = 0;     // Times the device did not give.
    std::string summary;
};

State state;

// An event recorded now on the default stream, or none when the device refuses.
hipEvent_t mark() {
    hipEvent_t event = nullptr;
    if (!state.spare.empty()) {
        event = state.spare.back();
        state.spare.pop_back();
    } else if (hipEventCreate(&event) != hipSuccess) {
        return nullptr;
    }
    if (hipEventRecord(event, nullptr) != hipSuccess) {
        state.spare.push_back(event);
        return nullptr;
    }
    return event;
}

void give_back(hipEvent_t event) {
    if (event != nullptr) {
        state.spare.push_back(event);
    }
}

// Milliseconds on the device between two reached events, or a negative number.
double between(hipEvent_t start, hipEvent_t stop) {
    float ms = 0;
    if (start == nullptr || stop == nullptr || hipEventElapsedTime(&ms, start, stop) != hipSuccess) {
        return -1;
    }
    return ms;
}

double median(std::vector<double>& values) {
    if (values.empty()) {
        return 0;
    }
    std::nth_element(values.begin(), values.begin() + values.size() / 2, values.end());
    return values[values.size() / 2];
}

// The device's time for a run of launches that compute nothing, each between a pair of events or bare. With pairs,
// the times between each pair are added to `inside`.
double launches(bool paired, std::vector<double>& inside) {
    std::vector<Pending> pairs;
    const hipEvent_t start = mark();
    for (int launch = 0; launch < kCalibrationLaunches; ++launch) {
        const hipEvent_t before = paired ? mark() : nullptr;
        launch_nothing();
        if (paired) {
            pairs.push_back({before, mark(), 0});
        }
    }
    const hipEvent_t stop = mark();
    if (stop != nullptr) {
        (void)hipEventSynchronize(stop);
    }
    const double whole = between(start, stop);
    for (const Pending& pair : pairs) {
        const double ms = between(pair.start, pair.stop);
        if (ms >= 0) {
            inside.push_back(ms);
        }
        give_back(pair.start);
        give_back(pair.stop);
    }
    give_back(start);
    give_back(stop);
    return whole;
}

// Measures what events cost: the time between a pair of events around a launch that computes nothing, and what such
// a pair adds to a run of launches for each launch. Medians over a few runs.
void calibrate() {
    std::vector<double> inside;
    std::vector<double> added;
    for (int round = 0; round < kCalibrationRounds; ++round) {
        const double bare = launches(false, inside);
        const double paired = launches(true, inside);
        if (bare >= 0 && paired >= 0) {
            added.push_back(std::max(paired - bare, 0.0) / kCalibrationLaunches);
        }
    }
    state.event_ms = median(inside);
    state.pair_ms = median(added);
}

std::string escaped(std::string_view text) {
    std::string out;
    for (const char c : text) {
        if (c == '"' || c == '\\') {
            out += '\\';
            out += c;
        } else if (static_cast<unsigned char>(c) < 0x20) {
            out += std::format("\\u{:04x}", static_cast<int>(c));
        } else {
            out += c;
        }
    }
    return out;
}

// The fields of a product and what it took, without braces. The phase leads in the summary.
std::string product_fields(const Key& key, const Stat& stat, bool with_phase) {
    std::string out;
    if (with_phase) {
        out += std::format(R"("phase":"{}",)", escaped(state.phases[key.phase].name));
    }
    out += std::format(R"("op":"{}","path":"{}",)", kOpNames[key.op], kPathNames[key.path]);
    if (key.reason != 0) {
        out += std::format(R"("reason":"{}",)", kReasonNames[key.reason]);
    }
    out += std::format(R"("m":{},"k":{},"n":{},"dtype":"{}","calls":{},"timed_calls":{},"ms":{:.4f},"min_ms":{:.4f},"max_ms":{:.4f})",
                       key.m, key.k, key.n, kDtypeNames[key.dtype], stat.calls, stat.timed, stat.ms,
                       stat.timed > 0 ? stat.low : 0.0, stat.high);
    return out;
}

void close_span() {
    const hipEvent_t stop = mark();
    if (stop != nullptr) {
        (void)hipEventSynchronize(stop);
    }
    roctxRangePop();
    const double wall = milliseconds(Clock::now() - state.wall);
    double gpu = between(state.start, stop);
    if (gpu < 0) {
        gpu = wall;
        ++state.unresolved;
    }
    for (const Pending& one : state.pending) {
        const double ms = between(one.start, one.stop);
        if (ms < 0) {
            ++state.unresolved;
        } else {
            state.groups[one.group].stat.add(std::max(ms - state.event_ms, 0.0));
        }
        give_back(one.start);
        give_back(one.stop);
    }
    give_back(state.start);
    give_back(stop);

    const int phase = static_cast<int>(state.phase);
    PhaseTotal& total = state.phases[state.phase];
    ++total.spans;
    total.rows += state.rows;
    total.tokens += state.tokens;
    double int4_ms = 0;
    double head_ms = 0;
    long long int4_timed = 0;
    std::string line = std::format(R"({{"record":"span","seq":{},"phase":"{}","rows":{},"tokens":{},"wall_ms":{:.4f},"gpu_ms":{:.4f},"timed":{},"products":[)",
                                   state.spans, escaped(total.name), state.rows, state.tokens, wall, gpu, state.span_timed);
    bool first = true;
    for (const Group& group : state.groups) {
        const Key key = key_of(phase, group.product);
        state.products[key].merge(group.stat);
        (group.product.op == Op::Head ? head_ms : int4_ms) += group.stat.ms;
        int4_timed += group.product.op == Op::Head ? 0 : group.stat.timed;
        line += first ? "{" : ",{";
        line += product_fields(key, group.stat, false);
        line += "}";
        first = false;
    }
    line += "]}\n";
    std::fputs(line.c_str(), state.file);
    if (state.span_timed) {
        ++total.timed_spans;
        total.timed_gpu_ms += gpu;
        total.timed_int4_ms += int4_ms;
        total.timed_head_ms += head_ms;
        total.timed_int4_calls += int4_timed;
    } else {
        total.wall_ms += wall;
        total.gpu_ms += gpu;
        total.head_ms += head_ms;
    }
    ++state.spans;
    state.pending.clear();
    state.groups.clear();
    state.open = false;
}

std::string summary_line() {
    std::string out = std::format(R"({{"record":"summary","elapsed_ms":{:.3f},"spans":{},"every":{},"event_ms":{:.4f},"pair_ms":{:.4f},"phases":[)",
                                  milliseconds(Clock::now() - state.began), state.spans, state.every, state.event_ms,
                                  state.pair_ms);
    for (std::size_t phase = 0; phase < state.phases.size(); ++phase) {
        const PhaseTotal& total = state.phases[phase];
        out += std::format(R"({}{{"phase":"{}","spans":{},"rows":{},"tokens":{},"wall_ms":{:.4f},"gpu_ms":{:.4f},"head_ms":{:.4f},)"
                           R"("timed_spans":{},"timed_gpu_ms":{:.4f},"timed_int4_ms":{:.4f},"timed_head_ms":{:.4f},"timed_int4_calls":{}}})",
                           phase == 0 ? "" : ",", escaped(total.name), total.spans, total.rows, total.tokens, total.wall_ms,
                           total.gpu_ms, total.head_ms, total.timed_spans, total.timed_gpu_ms, total.timed_int4_ms,
                           total.timed_head_ms, total.timed_int4_calls);
    }
    out += R"(],"products":[)";
    bool first = true;
    for (const auto& [key, stat] : state.products) {
        out += first ? "{" : ",{";
        out += product_fields(key, stat, true);
        out += "}";
        first = false;
    }
    long long fallback_calls = 0;
    for (const long long count : state.fallbacks) {
        fallback_calls += count;
    }
    out += std::format(R"(],"counters":{{"kernel_calls":{},"head_calls":{},"fallback_calls":{})", state.kernel_calls,
                       state.head_calls, fallback_calls);
    for (int reason = 1; reason < kReasons; ++reason) {
        out += std::format(R"(,"fallback_{}":{})", kReasonNames[reason], state.fallbacks[reason]);
    }
    out += std::format(R"(,"outside_calls":{},"unresolved":{}}}}})", state.outside_calls, state.unresolved);
    return out;
}

template <std::size_t N>
bool within(int value, const std::array<std::string_view, N>&) {
    return value >= 0 && value < static_cast<int>(N);
}

}  // namespace

bool on() { return state.on; }

void begin(const Product& product) {
    if (!state.on || state.depth++ > 0) {
        return;
    }
    if (product.op == Op::Head) {
        ++state.head_calls;
    } else if (product.path == Path::Unpack) {
        ++state.fallbacks[static_cast<int>(product.reason)];
    } else {
        ++state.kernel_calls;
    }
    state.timed = false;
    if (!state.open) {
        ++state.outside_calls;
        return;
    }
    const Key key = key_of(0, product);
    std::size_t index = 0;
    while (index < state.groups.size() && key_of(0, state.groups[index].product) != key) {
        ++index;
    }
    if (index == state.groups.size()) {
        state.groups.push_back({product, {}, std::format("{} {} {}x{}x{} {}", kOpNames[key.op], kPathNames[key.path], key.m, key.k, key.n, kDtypeNames[key.dtype])});
    }
    ++state.groups[index].stat.calls;
    // The head is one product of a span and is always timed.
    if (!state.span_timed && product.op != Op::Head) {
        return;
    }
    state.timed = true;
    state.product_group = index;
    roctxRangePushA(state.groups[index].label.c_str());
    state.product_start = mark();
}

void end() {
    if (!state.on || state.depth == 0 || --state.depth > 0 || !state.timed) {
        return;
    }
    const hipEvent_t stop = mark();
    roctxRangePop();
    state.pending.push_back({state.product_start, stop, state.product_group});
    state.timed = false;
}

}  // namespace rdna4_w4a16::dev

extern "C" {

using namespace rdna4_w4a16::dev;

// Starts the developer mode and its record file. One span in every `every` of a phase has its int4 products timed,
// none when it is zero. Returns 0, -1 when the file cannot be written, -2 when the mode is on.
RDNA4_W4A16_API int rdna4_w4a16_dev_start(const char* path, int every) {
    if (state.on) {
        return -2;
    }
    state = State{};
    state.file = std::fopen(path, "w");
    if (state.file == nullptr) {
        return -1;
    }
    state.every = every > 0 ? every : 0;
    int device = 0;
    hipDeviceProp_t properties{};
    std::string name = "unknown";
    std::string arch;
    if (hipGetDevice(&device) == hipSuccess && hipGetDeviceProperties(&properties, device) == hipSuccess) {
        name = properties.name;
        arch = properties.gcnArchName;
        // The architecture name carries the device's feature flags after a colon.
        arch = arch.substr(0, arch.find(':'));
    }
    (void)hipDeviceSynchronize();
    calibrate();
    const auto now = std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now());
    const std::string line = std::format(R"({{"record":"run","schema":{},"started":"{:%FT%TZ}","device":"{}","arch":"{}","every":{},"event_ms":{:.4f},"pair_ms":{:.4f}}})",
                                         kSchema, now, escaped(name), escaped(arch), state.every, state.event_ms, state.pair_ms);
    std::fputs((line + "\n").c_str(), state.file);
    state.began = Clock::now();
    state.on = true;
    return 0;
}

// Opens a span: a piece of the caller's work, named `name`, over `rows` rows of `tokens` tokens in all. Spans of one
// name form a phase. Returns 0, or -1 when the mode is off, the name is empty or a span is open.
RDNA4_W4A16_API int rdna4_w4a16_dev_span_begin(const char* name, long long rows, long long tokens) {
    if (!state.on || state.open || name == nullptr || *name == '\0') {
        return -1;
    }
    std::size_t phase = 0;
    while (phase < state.phases.size() && state.phases[phase].name != name) {
        ++phase;
    }
    if (phase == state.phases.size()) {
        state.phases.push_back({});
        state.phases.back().name = name;
    }
    state.open = true;
    state.phase = phase;
    state.rows = rows;
    state.tokens = tokens;
    // The second span of a phase is the first timed one, so a phase of a few spans has both kinds.
    state.span_timed = state.every > 0 && state.phases[phase].spans % state.every == 1 % state.every;
    roctxRangePushA(state.phases[phase].name.c_str());
    state.wall = Clock::now();
    state.start = mark();
    return 0;
}

// Closes the open span once the device has reached its end, and writes its line.
RDNA4_W4A16_API int rdna4_w4a16_dev_span_end(void) {
    if (!state.on || !state.open) {
        return -1;
    }
    close_span();
    return 0;
}

// The start of a product the caller runs itself: the checkpoint's own path of an int4 layer, or the head.
RDNA4_W4A16_API int rdna4_w4a16_dev_op_begin(int op, int path, int reason, int m, int k, int n, int dtype) {
    if (!within(op, kOpNames) || !within(path, kPathNames) || !within(reason, kReasonNames) || !within(dtype, kDtypeNames)) {
        return -1;
    }
    begin({static_cast<Op>(op), static_cast<Path>(path), static_cast<Reason>(reason), m, k, n, static_cast<Dtype>(dtype)});
    return 0;
}

RDNA4_W4A16_API int rdna4_w4a16_dev_op_end(void) {
    end();
    return 0;
}

// Ends the developer mode: closes an open span, writes the summary as the file's last line and returns it. The text
// stays valid until the mode starts again. Empty when the mode is off.
RDNA4_W4A16_API const char* rdna4_w4a16_dev_finish(void) {
    if (!state.on) {
        return "";
    }
    if (state.open) {
        close_span();
    }
    state.summary = summary_line();
    std::fputs((state.summary + "\n").c_str(), state.file);
    std::fclose(state.file);
    state.file = nullptr;
    for (const hipEvent_t event : state.spare) {
        (void)hipEventDestroy(event);
    }
    state.spare.clear();
    state.on = false;
    return state.summary.c_str();
}

}  // extern "C"
