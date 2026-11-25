import Foundation

private let gib = Int64(1024 * 1024 * 1024)
private var failures = 0

private func expect(_ condition: @autoclosure () -> Bool, _ message: String) {
    if condition() {
        print("PASS: \(message)")
    } else {
        print("FAIL: \(message)")
        failures += 1
    }
}

private func close(_ lhs: Double, _ rhs: Double, tolerance: Double = 0.0001) -> Bool {
    abs(lhs - rhs) <= tolerance
}

@main
struct MemoryBudgetTests {
    static func main() {
let budget = MemoryBudget(totalBytes: 100 * gib,
                          budgetBytes: 80 * gib,
                          weightsBytes: 30 * gib,
                          cacheLimitBytes: 60 * gib,
                          isLoaded: false,
                          kvBytesPerToken: 1024 * 1024,
                          modelContext: 262_144)

// The cache limit must not move as a side effect of changing Context.
let cacheLimit = budget.maxCacheBytes(contextTokens: 0, bits: 0)
expect(cacheLimit == 57 * gib, "cache limit is the fixed 95% cache budget")
let cacheAtStop = budget.clampCache(cacheLimit, contextTokens: 0, bits: 0)
let cachePastStop = budget.clampCache(cacheLimit + 20 * gib, contextTokens: 0, bits: 0)
expect(cacheAtStop == cachePastStop, "cache value stops changing after the slider stop")
let contextAtCacheStop = budget.effectiveContext(requested: 262_144,
                                                 cacheBytes: cacheAtStop, bits: 0)
let contextPastCacheStop = budget.effectiveContext(requested: 262_144,
                                                   cacheBytes: cachePastStop, bits: 0)
expect(contextAtCacheStop == contextPastCacheStop,
       "context segment is unchanged when cache is dragged past its stop")
let selectedContextGB = budget.contextBytes(tokens: 8192, bits: 0) / Double(gib)
let lowCacheMap = MemoryMapLayout(totalGB: 100, weightsGB: 30, cacheGB: 1,
                                  contextGB: selectedContextGB, nonLLMGB: 10)
let highCacheMap = MemoryMapLayout(totalGB: 100, weightsGB: 30, cacheGB: 20,
                                   contextGB: selectedContextGB, nonLLMGB: 10)
expect(lowCacheMap.widths(totalWidth: 1000).context
       == highCacheMap.widths(totalWidth: 1000).context,
       "yellow segment depends on selected Context, not Prompt cache")
expect(budget.effectiveContext(requested: 262_144, cacheBytes: 16 * gib, bits: 0)
       != budget.effectiveContext(requested: 262_144, cacheBytes: 57 * gib, bits: 0),
       "effective context remains a separate cache trade-off forecast")

// Context must clamp independently of the requested value beyond its RAM limit.
let contextLimit = budget.maxContext(cacheBytes: 0, bits: 0)
let contextAtStop = budget.clampRequestedContext(contextLimit + 100_000, bits: 0)
let contextPastStop = budget.clampRequestedContext(contextLimit + 200_000, bits: 0)
expect(contextAtStop == contextPastStop,
       "context value stops changing after the slider stop")
expect(budget.effectiveContext(requested: contextAtStop, cacheBytes: 10 * gib, bits: 0)
       == budget.effectiveContext(requested: contextPastStop, cacheBytes: 10 * gib, bits: 0),
       "context graph segment is unchanged after the slider stop")
expect(contextLimit < 262_144, "test budget has a real context boundary")
let contextRightInitial = budget.maxContext(cacheBytes: 0, bits: 0)
let highContextCacheMax = budget.maxCacheBytes(contextTokens: contextRightInitial, bits: 0)
let lowContextCacheMax = budget.maxCacheBytes(contextTokens: 4096, bits: 0)
expect(lowContextCacheMax > highContextCacheMax,
       "moving Context left increases the Prompt cache maximum")
let contextRightAgain = budget.maxContext(cacheBytes: 0, bits: 0)
expect(contextRightAgain == contextRightInitial,
       "moving Context left and right does not shrink its maximum")
expect(budget.clampCache(lowContextCacheMax + 20 * gib,
                         contextTokens: 4096, bits: 0) == lowContextCacheMax,
       "newly freed Prompt cache range still has a hard stop")
let cacheBeforeContextDrag = budget.maxCacheBytes(contextTokens: 16_384, bits: 0)
let contextAfterDrag = budget.clampRequestedContext(262_144, bits: 0)
let cacheAfterContextDrag = budget.clampCache(cacheBeforeContextDrag,
                                              contextTokens: contextAfterDrag, bits: 0)
expect(contextAfterDrag == contextRightInitial,
       "Context reaches its maximum in one drag even with full Prompt cache")
expect(cacheAfterContextDrag < cacheBeforeContextDrag,
       "Context drag automatically releases Prompt cache memory")

// A loaded model's weights are already represented in available RAM.
let loaded = MemoryBudget(totalBytes: 100 * gib, budgetBytes: 80 * gib,
                          weightsBytes: 30 * gib, cacheLimitBytes: 60 * gib,
                          isLoaded: true, kvBytesPerToken: 1024 * 1024,
                          modelContext: 262_144)
expect(loaded.contextLimit(cacheBytes: 10 * gib, bits: 0)
       > budget.contextLimit(cacheBytes: 10 * gib, bits: 0),
       "loaded-model budget does not subtract weights twice")

// The bar always uses the full width, with free space between LLM and non-LLM.
let map = MemoryMapLayout(totalGB: 100, weightsGB: 20, cacheGB: 10,
                          contextGB: 5, nonLLMGB: 15)
let widths = map.widths(totalWidth: 1000)
expect(close(Double(widths.weights + widths.cache + widths.context + widths.free + widths.nonLLM), 1000),
       "memory map occupies exactly 100% width")
expect(close(Double(widths.weights), 200) && close(Double(widths.cache), 100)
       && close(Double(widths.context), 50) && close(Double(widths.nonLLM), 150),
       "LLM and non-LLM segment widths follow the physical RAM scale")
expect(close(Double(widths.free), 500), "gray area is the remaining memory")

// Oversubscription cannot push the red segment outside the bar.
let overflow = MemoryMapLayout(totalGB: 100, weightsGB: 70, cacheGB: 50,
                               contextGB: 20, nonLLMGB: 30).widths(totalWidth: 1000)
expect(close(Double(overflow.weights + overflow.cache + overflow.context
                    + overflow.free + overflow.nonLLM), 1000),
       "oversubscribed map remains inside 100% width")
expect(overflow.free == 0 && overflow.nonLLM > 0,
       "oversubscribed map removes gray space but keeps non-LLM visible")

if failures > 0 { exit(1) }
print("All memory budget tests passed")
    }
}
