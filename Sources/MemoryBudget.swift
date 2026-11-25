import Foundation

struct MemoryBudget {
    let totalBytes: Int64
    let budgetBytes: Int64
    let weightsBytes: Int64
    let cacheLimitBytes: Int64
    let isLoaded: Bool
    let kvBytesPerToken: Double
    let modelContext: Int?

    private static let gib = Int64(1024 * 1024 * 1024)

    private var allocatedWeightsBytes: Int64 {
        isLoaded ? 0 : weightsBytes
    }

    private var reserveBytes: Int64 {
        max(2 * Self.gib, Int64(Double(max(0, budgetBytes)) * 0.10))
    }

    var baseMaxCacheBytes: Int64 {
        max(0, min(Int64(Double(totalBytes) * 0.75),
                   Int64(Double(cacheLimitBytes) * 0.95)))
    }

    func maxCacheBytes(contextTokens: Int, bits: Int) -> Int64 {
        max(0, baseMaxCacheBytes - Int64(contextBytes(tokens: contextTokens, bits: bits)))
    }

    func clampCache(_ bytes: Int64, contextTokens: Int, bits: Int) -> Int64 {
        min(max(0, bytes), maxCacheBytes(contextTokens: contextTokens, bits: bits))
    }

    private func scaledKV(bits: Int) -> Double {
        Double(kvBytesPerToken) * Double(bits == 0 ? 16 : bits) / 16
    }

    func contextBytes(tokens: Int, bits: Int) -> Double {
        Double(tokens) * scaledKV(bits: bits)
    }

    func contextLimit(cacheBytes: Int64, bits: Int) -> Int {
        guard kvBytesPerToken > 0 else { return 262_144 }
        let remaining = max(0, budgetBytes - reserveBytes - allocatedWeightsBytes
                            - min(max(0, cacheBytes), baseMaxCacheBytes))
        return Int(Double(remaining) / scaledKV(bits: bits))
    }

    func maxContext(cacheBytes: Int64, bits: Int) -> Int {
        min(262_144, max(4096, contextLimit(cacheBytes: cacheBytes, bits: bits)))
    }

    func clampContext(_ requested: Int, cacheBytes: Int64, bits: Int) -> Int {
        min(max(4096, requested), maxContext(cacheBytes: cacheBytes, bits: bits))
    }

    func clampRequestedContext(_ requested: Int, bits: Int) -> Int {
        min(max(4096, requested), maxContext(cacheBytes: 0, bits: bits))
    }

    func effectiveContext(requested: Int, cacheBytes: Int64, bits: Int) -> Int {
        let cap = min(requested, modelContext ?? requested,
                      contextLimit(cacheBytes: cacheBytes, bits: bits))
        guard kvBytesPerToken > 0 else { return cap }
        let cacheBudget = Int(Double(min(max(0, cacheBytes), baseMaxCacheBytes)) / scaledKV(bits: bits))
        let limit = min(cap, cacheBudget)
        return (modelContext ?? 4096) >= 4096 && limit < 4096 ? 0 : limit
    }
}

struct MemoryMapLayout {
    let totalGB: Double
    let weightsGB: Double
    let cacheGB: Double
    let contextGB: Double
    let nonLLMGB: Double

    func widths(totalWidth: CGFloat) -> (weights: CGFloat, cache: CGFloat,
                                         context: CGFloat, free: CGFloat, nonLLM: CGFloat) {
        guard totalWidth > 0, totalGB > 0 else { return (0, 0, 0, 0, 0) }
        let values = [max(0, weightsGB), max(0, cacheGB), max(0, contextGB), max(0, nonLLMGB)]
        let sum = values.reduce(0, +)
        let scale = sum > totalGB ? totalGB / sum : 1
        let freeGB = max(0, totalGB - sum)
        return (totalWidth * CGFloat(values[0] * scale / totalGB),
                totalWidth * CGFloat(values[1] * scale / totalGB),
                totalWidth * CGFloat(values[2] * scale / totalGB),
                totalWidth * CGFloat(freeGB / totalGB),
                totalWidth * CGFloat(values[3] * scale / totalGB))
    }
}
