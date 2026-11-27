import AppKit
import SwiftUI

struct ModelRating: Decodable {
    let score: Double
    let quality: Double
    let speed: Double
    let fit: Double
    let context: Double
    let tps: Double
    let mem: Double
    let params: Double
    let useCase: String
}

struct ModelInfo: Decodable, Identifiable {
    let id: String
    let installed: Bool
    let size: String
    let sizeBytes: Int64
    var downloading: Bool
    var downloadError: String?
    var downloadBytes: Int64
    var downloadTotal: Int64
    var downloadRate: Double
    var downloadPaused: Bool
    let downloads: Int
    let likes: Int
    let score: Double
    let rating: ModelRating?
    let ramGb: Double?
    let parametersB: Double?
    let activeParametersB: Double?
    let context: String?
    let tools: Bool
    let gguf: Bool
    let recommended: Bool
    let helper: Bool
    let fits: Bool?
    let deletedBefore: Bool
    var hfURL: URL { URL(string: "https://huggingface.co/\(id)")! }

    mutating func apply(_ progress: DownloadProgress) {
        downloading = progress.active
        downloadError = progress.error
        downloadBytes = progress.bytes
        downloadTotal = progress.total
        downloadRate = progress.rate
        downloadPaused = progress.paused
    }
}

struct ModelCatalogResponse: Decodable {
    let ramGb: Int
    let gpu: String
    let recommended: String
    let smallModel: String
    let downloaded: [ModelInfo]
    let models: [ModelInfo]
    let hasMore: Bool
    let total: Int
    let error: String?
}

struct DownloadProgress: Decodable {
    let active: Bool
    let error: String?
    let bytes: Int64
    let total: Int64
    let rate: Double
    let paused: Bool
}

struct LaunchParameters: Decodable {
    var maxCtx = 110_000
    var promptCacheBytes = 16 * 1024 * 1024 * 1024
    var kvBits = 0
    var kvGroupSize = 64
    var loadTimeout = 900
    var clientTimeout = 1800
    var temperature = 0.0
    var topP = 1.0
    var topK = 0
    var minP = 0.0
    var maxTokens = 32768
}

struct ConfigPreview: Decodable {
    let files: [String]
    let backupRoot: String
}

struct LaunchSettings: Decodable {
    let settings: LaunchParameters
    let defaults: LaunchParameters
    let kvBytesPerToken: Double?
    let effectiveCtx: Int?
    let modelCtx: Int?
    let availableBytes: Int64
    let reclaimableBytes: Int64
    let isLoaded: Bool
    let budgetAvailableBytes: Int64
    let weightsBytes: Int64
    let cacheLimitBytes: Int64
    let configPreview: ConfigPreview
}

enum CatalogSort: String, CaseIterable, Identifiable {
    case fit, dl
    var id: String { rawValue }
    var title: String { self == .fit ? "By score" : "By downloads" }
}

enum CatalogScope: String, CaseIterable, Identifiable {
    case community, all
    var id: String { rawValue }
    var title: String { self == .community ? "mlx-community" : "All authors" }
}

enum CatalogFormat: String, CaseIterable, Identifiable {
    case mlx, gguf
    var id: String { rawValue }
    var title: String { rawValue.uppercased() }
}

@MainActor
final class ModelManagerModel: ObservableObject {
    @Published private(set) var downloaded: [ModelInfo] = []
    @Published private(set) var models: [ModelInfo] = []
    @Published private(set) var ramGb = 0
    @Published private(set) var gpu = "Apple Silicon"
    @Published private(set) var loading = false
    @Published private(set) var loadingMore = false
    @Published private(set) var deleting: String?
    @Published private(set) var startingDownloads: Set<String> = []
    @Published private(set) var hasMore = false
    @Published private(set) var total = 0
    @Published private(set) var error: String?
    @Published var sortMode = CatalogSort.fit
    @Published var scope = CatalogScope.community
    @Published var format = CatalogFormat.mlx
    @Published var hoveredModelID: String?
    @Published var query = ""
    @Published var selected: String?
    @Published var pendingDelete: ModelInfo?
    @Published var pendingLaunch: ModelInfo?
    @Published var pendingCancel: ModelInfo?
    @Published var launchDraft = LaunchParameters()
    @Published private(set) var loadingLaunch = false
    @Published private(set) var launchSettingsLoaded = false
    @Published private(set) var launchDefaults = LaunchParameters()
    @Published private(set) var launchKVBytesPerToken: Double?
    @Published private(set) var launchModelContext: Int?
    @Published private(set) var launchAvailableBytes: Int64 = 0
    @Published private(set) var launchCurrentAvailableBytes: Int64 = 0
    @Published private(set) var launchReclaimableBytes: Int64 = 0
    @Published private(set) var launchIsLoaded = false
    @Published private(set) var launchWeightsBytes: Int64 = 0
    @Published private(set) var launchCacheLimitBytes: Int64 = 0
    @Published private(set) var launchConfigPreview = ConfigPreview(files: [], backupRoot: "")
    @Published var approveConfigWrites = false
    @Published private(set) var savingLaunch = false
    @Published var launchError: String?
    @Published var downloadedHeight = 218.0
    private var splitDragStart: CGFloat?
    private var resettingFilters = false

    func resizeSplit(_ translation: CGFloat, totalHeight: CGFloat) {
        if splitDragStart == nil { splitDragStart = downloadedHeight }
        let upper = max(100, totalHeight - 360)
        downloadedHeight = min(upper, max(100, (splitDragStart ?? 218) + translation))
    }

    func clampDownloadedHeight(totalHeight: CGFloat) {
        downloadedHeight = min(downloadedHeight, max(100, totalHeight - 360))
    }

    func endSplitDrag() {
        splitDragStart = nil
    }

    func filtersChanged() {
        guard !resettingFilters else { return }
        Task { await reload() }
    }

    func nudgeSplit(_ delta: CGFloat, totalHeight: CGFloat) {
        splitDragStart = downloadedHeight
        resizeSplit(delta, totalHeight: totalHeight)
        endSplitDrag()
    }

    private let port: Int
    private let session: URLSession
    private var generation = 0
    private var downloadPollTask: Task<Void, Never>?

    init(port: Int) {
        self.port = port
        let cfg = URLSessionConfiguration.ephemeral
        cfg.timeoutIntervalForRequest = 180
        cfg.timeoutIntervalForResource = 180
        session = URLSession(configuration: cfg)
    }

    func search(_ text: String) async {
        query = text.trimmingCharacters(in: .whitespacesAndNewlines)
        await reload()
    }

    func resetFilters() async {
        resettingFilters = true
        query = ""
        scope = .community
        format = .mlx
        sortMode = .fit
        resettingFilters = false
        await reload()
    }

    @Published private(set) var hasLoadedCatalog = false

    func loadInitialIfNeeded() async {
        guard !hasLoadedCatalog, !loading else { return }
        await reload()
    }

    func reload(refresh: Bool = false, background: Bool = false) async {
        generation += 1
        let request = generation
        await fetch(offset: 0, append: false, refresh: false,
                    request: request, localOnly: true, background: background)
        guard request == generation else { return }
        await fetch(offset: 0, append: false, refresh: refresh,
                    request: request, background: background)
    }

    func loadMore() async {
        guard hasMore, !loading, !loadingMore else { return }
        generation += 1
        await fetch(offset: models.count, append: true, refresh: false,
                    request: generation)
    }

    func delete(_ model: String) async {
        guard deleting == nil else { return }
        deleting = model
        defer { deleting = nil }
        do {
            try await post("/delete", model: model)
            await reload()
        } catch {
            self.error = error.localizedDescription
        }
    }

    func download(_ item: ModelInfo) async {
        let model = item.id
        guard !startingDownloads.contains(model) else { return }
        startingDownloads.insert(model)
        defer { startingDownloads.remove(model) }
        do {
            try await post("/download", model: model)
            if let index = models.firstIndex(where: { $0.id == model }) {
                var local = models.remove(at: index)
                local.downloading = true
                if !downloaded.contains(where: { $0.id == model }) {
                    downloaded.insert(local, at: 0)
                }
            }
            _ = await refreshDownloadProgress()
            startDownloadPolling()
            await reload(background: true)
        } catch {
            self.error = error.localizedDescription
        }
    }

    func pauseDownload(_ model: String, resume: Bool) async {
        do {
            try await post(resume ? "/resume" : "/pause", model: model)
            _ = await refreshDownloadProgress()
        } catch {
            self.error = error.localizedDescription
        }
    }

    func cancelDownload(_ item: ModelInfo) {
        pendingCancel = item
    }

    func confirmCancelDownload(_ model: String) async {
        do {
            try await post("/cancel", model: model)
            _ = await refreshDownloadProgress()
            await reload(background: true)
        } catch {
            self.error = error.localizedDescription
        }
    }

    private func startDownloadPolling() {
        guard downloadPollTask == nil else { return }
        downloadPollTask = Task {
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(1))
                if Task.isCancelled { break }
                if await refreshDownloadProgress() { continue }
                await reload(background: true)
                break
            }
            downloadPollTask = nil
        }
    }

    private func refreshDownloadProgress() async -> Bool {
        guard let url = URL(string: "http://127.0.0.1:\(port)/downloads") else { return false }
        do {
            let (data, _) = try await session.data(from: url)
            let decoder = JSONDecoder()
            decoder.keyDecodingStrategy = .convertFromSnakeCase
            let jobs = try decoder.decode([String: DownloadProgress].self, from: data)
            for (id, job) in jobs {
                if let index = downloaded.firstIndex(where: { $0.id == id }) {
                    downloaded[index].apply(job)
                }
                if let index = models.firstIndex(where: { $0.id == id }) {
                    models[index].apply(job)
                }
            }
            return jobs.values.contains(where: { $0.active })
        } catch {
            return true  // transient backend restart: retry without dropping rows
        }
    }

    private func fetch(offset: Int, append: Bool, refresh: Bool,
                       request: Int, localOnly: Bool = false,
                       background: Bool = false) async {
        loading = !append && !background
        loadingMore = append
        if !append { error = nil }
        defer {
            if request == generation {
                loading = false
                loadingMore = false
            }
        }
        var components = URLComponents(string: "http://127.0.0.1:\(port)/models")
        components?.queryItems = [
            URLQueryItem(name: "scope", value: scope.rawValue),
            URLQueryItem(name: "format", value: format.rawValue),
            URLQueryItem(name: "sort", value: sortMode.rawValue),
            URLQueryItem(name: "query", value: query),
            URLQueryItem(name: "offset", value: String(offset)),
            URLQueryItem(name: "limit", value: "50"),
            URLQueryItem(name: "refresh", value: refresh ? "1" : "0"),
            URLQueryItem(name: "local_only", value: localOnly ? "1" : "0"),
        ]
        guard let url = components?.url else { return }
        do {
            let (data, response) = try await session.data(from: url)
            try Self.requireSuccess(response, data: data)
            let decoder = JSONDecoder()
            decoder.keyDecodingStrategy = .convertFromSnakeCase
            let payload = try decoder.decode(ModelCatalogResponse.self, from: data)
            guard request == generation else { return }
            downloaded = payload.downloaded
            if !background || !localOnly {
                models = append ? models + payload.models : payload.models
                hasMore = payload.hasMore
                total = payload.total
            }
            ramGb = payload.ramGb
            gpu = payload.gpu
            error = payload.error
            if !localOnly { hasLoadedCatalog = true }
            if downloaded.contains(where: { $0.downloading })
                || models.contains(where: { $0.downloading }) {
                startDownloadPolling()
            }
        } catch {
            guard request == generation else { return }
            self.error = error.localizedDescription
            if !append && (!background || !localOnly) { models = [] }
        }
    }

    func beginLaunch(_ item: ModelInfo) {
        launchError = nil
        launchSettingsLoaded = false
        approveConfigWrites = false
        pendingLaunch = item
    }

    func loadLaunchSettings(_ item: ModelInfo) async {
        loadingLaunch = true
        launchSettingsLoaded = false
        launchError = nil
        defer { loadingLaunch = false }
        var components = URLComponents(string: "http://127.0.0.1:\(port)/settings")
        components?.queryItems = [URLQueryItem(name: "model", value: item.id)]
        guard let url = components?.url else { return }
        do {
            let (data, response) = try await session.data(from: url)
            try Self.requireSuccess(response, data: data)
            let decoder = JSONDecoder()
            decoder.keyDecodingStrategy = .convertFromSnakeCase
            let settings = try decoder.decode(LaunchSettings.self, from: data)
            launchDraft = settings.settings
            launchDefaults = settings.defaults
            launchKVBytesPerToken = settings.kvBytesPerToken
            launchModelContext = settings.modelCtx
            launchAvailableBytes = settings.budgetAvailableBytes
            launchCurrentAvailableBytes = settings.availableBytes
            launchReclaimableBytes = settings.reclaimableBytes
            launchIsLoaded = settings.isLoaded
            launchWeightsBytes = settings.weightsBytes
            launchCacheLimitBytes = settings.cacheLimitBytes
            launchConfigPreview = settings.configPreview
            clampLaunchDraft()
            launchSettingsLoaded = true
        } catch {
            launchError = error.localizedDescription
        }
    }

    func resetLaunchSettings() {
        launchDraft = launchDefaults
        clampLaunchDraft()
    }

    func clampLaunchDraft() {
        launchDraft.maxCtx = max(4096, min(launchDraft.maxCtx, maxLaunchContext))
        launchDraft.promptCacheBytes = Int(launchMemoryBudget.clampCache(
            Int64(launchDraft.promptCacheBytes), contextTokens: launchDraft.maxCtx,
            bits: launchDraft.kvBits))
        launchDraft.maxTokens = max(256, min(launchDraft.maxTokens, effectiveLaunchContext))
        launchDraft.temperature = min(1, launchDraft.temperature)
    }

    func saveAndLoad(_ item: ModelInfo, panel: PanelModel) async {
        guard !savingLaunch, !loadingLaunch, launchSettingsLoaded,
              approveConfigWrites,
              let url = URL(string: "http://127.0.0.1:\(port)/settings") else { return }
        savingLaunch = true
        defer { savingLaunch = false }
        do {
            var request = URLRequest(url: url)
            request.httpMethod = "POST"
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            request.httpBody = try JSONSerialization.data(withJSONObject: [
                "model": item.id,
                "max_ctx": launchDraft.maxCtx,
                "prompt_cache_bytes": launchDraft.promptCacheBytes,
                "kv_bits": launchDraft.kvBits,
                "kv_group_size": launchDraft.kvGroupSize,
                "load_timeout": launchDraft.loadTimeout,
                "client_timeout": launchDraft.clientTimeout,
                "temperature": launchDraft.temperature,
                "top_p": launchDraft.topP,
                "top_k": launchDraft.topK,
                "min_p": launchDraft.minP,
                "max_tokens": min(launchDraft.maxTokens, effectiveLaunchContext),
            ])
            let (data, response) = try await session.data(for: request)
            try Self.requireSuccess(response, data: data)
            pendingLaunch = nil
            panel.runModel(item.id, configWriteConsent: true)
        } catch {
            launchError = error.localizedDescription
        }
    }

    var launchMemoryBudget: MemoryBudget {
        MemoryBudget(totalBytes: Int64(Double(ramGb) * 1024 * 1024 * 1024),
                     budgetBytes: launchAvailableBytes,
                     weightsBytes: launchWeightsBytes,
                     cacheLimitBytes: launchCacheLimitBytes,
                     isLoaded: launchIsLoaded,
                     kvBytesPerToken: launchKVBytesPerToken ?? 0,
                     modelContext: launchModelContext)
    }

    var launchRAMContextLimit: Int? {
        guard launchKVBytesPerToken ?? 0 > 0 else { return nil }
        return launchMemoryBudget.contextLimit(cacheBytes: Int64(launchDraft.promptCacheBytes),
                                               bits: launchDraft.kvBits)
    }

    var maxLaunchContext: Int {
        launchMemoryBudget.maxContext(cacheBytes: 0, bits: launchDraft.kvBits)
    }

    var maxCacheGB: Double {
        Double(launchMemoryBudget.maxCacheBytes(contextTokens: launchDraft.maxCtx,
                                                bits: launchDraft.kvBits)) / 1024 / 1024 / 1024
    }

    var effectiveLaunchContext: Int {
        launchMemoryBudget.effectiveContext(requested: launchDraft.maxCtx,
                                            cacheBytes: Int64(launchDraft.promptCacheBytes),
                                            bits: launchDraft.kvBits)
    }

    private func post(_ path: String, model: String) async throws {
        guard let url = URL(string: "http://127.0.0.1:\(port)\(path)") else { return }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: ["model": model])
        let (data, response) = try await session.data(for: request)
        try Self.requireSuccess(response, data: data)
    }

    private static func requireSuccess(_ response: URLResponse, data: Data) throws {
        let status = (response as? HTTPURLResponse)?.statusCode ?? 0
        guard !(200..<300).contains(status) else { return }
        struct ErrorPayload: Decodable { let error: String }
        let message = (try? JSONDecoder().decode(ErrorPayload.self, from: data).error)
            ?? "Request failed with status \(status)"
        throw NSError(domain: "MLXStack", code: status,
                      userInfo: [NSLocalizedDescriptionKey: message])
    }
}

struct ModelManagerView: View {
    @ObservedObject var model: ModelManagerModel
    @ObservedObject var panel: PanelModel
    @ObservedObject var navigation: AppNavigation

    private var selectedModel: ModelInfo? {
        (model.downloaded + model.models).first { $0.id == model.selected }
    }

    private var visibleDownloaded: [ModelInfo] {
        let query = model.query.trimmingCharacters(in: .whitespacesAndNewlines)
        return model.downloaded.filter { query.isEmpty || $0.id.localizedCaseInsensitiveContains(query) }
    }

    var body: some View {
        GeometryReader { geometry in
            ZStack {
            VisualEffectBackground()
            Color.black.opacity(0.62)
            VStack(alignment: .leading, spacing: 12) {
                header
                controls
                if let error = model.error {
                    ErrorBanner(message: error)
                }
                if let error = panel.actionError {
                    ErrorBanner(message: error) { panel.actionError = nil }
                }
                downloadedBlock
                listSplitter(totalHeight: geometry.size.height)
                catalogBlock
                ratingBreakdown
            }
            .padding(18)
        }
        .frame(width: geometry.size.width, height: geometry.size.height)
        .onAppear { model.clampDownloadedHeight(totalHeight: geometry.size.height) }
        .onChange(of: geometry.size.height) { _, height in
            model.clampDownloadedHeight(totalHeight: height)
        }
        .task(id: panel.everConnected) {
            if panel.everConnected { await model.loadInitialIfNeeded() }
        }
        .alert(item: $model.pendingDelete) { item in
            Alert(title: Text("Delete downloaded model?"),
                  message: Text("\(item.id)\n\(item.size) will be removed from disk."
                      + (item.downloading ? "\nAnother process is still downloading it." : "")),
                  primaryButton: .destructive(Text("Delete")) {
                      Task { await model.delete(item.id) }
                  }, secondaryButton: .cancel())
        }
        .alert(item: $model.pendingCancel) { item in
            Alert(title: Text("Cancel this download?"),
                  message: Text("The partially downloaded \(item.id) files may be removed. You can start again later from Hugging Face."),
                  primaryButton: .destructive(Text("Cancel Download")) {
                      Task { await model.confirmCancelDownload(item.id) }
                  }, secondaryButton: .cancel(Text("Keep Downloading")))
        }
        .sheet(item: $model.pendingLaunch) { item in
            LaunchParametersView(item: item, model: model, panel: panel)
        }
    }
}

    private var header: some View {
        HStack(alignment: .firstTextBaseline) {
            ScreenPicker(navigation: navigation)
            Spacer()
            Text("\(model.gpu) · \(model.ramGb) GB RAM")
                .font(mono(12))
                .foregroundStyle(C.dim)
            Button {
                Task { await model.reload(refresh: true) }
            } label: {
                if model.loading {
                    ProgressView().controlSize(.small)
                } else {
                    Label("Refresh", systemImage: "arrow.clockwise")
                }
            }
            .disabled(model.loading || model.loadingMore)
            .help("Refresh downloaded models and Hugging Face rankings")
            .accessibilityLabel("Refresh model catalog")
        }
    }

    private var controls: some View {
        HStack(spacing: 10) {
            ZStack(alignment: .trailing) {
                TextField("Search Hugging Face models", text: $model.query)
                    .textFieldStyle(.roundedBorder)
                    .font(.system(size: 13))
                    .onSubmit { Task { await model.search(model.query) } }
                    .accessibilityLabel("Search Hugging Face models")
                if !model.query.isEmpty {
                    Button {
                        model.query = ""
                        Task { await model.search("") }
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                            .foregroundStyle(C.dim)
                    }
                    .buttonStyle(.plain)
                    .padding(.trailing, 6)
                    .help("Clear search")
                    .accessibilityLabel("Clear model search")
                }
            }
            .frame(minWidth: 260, maxWidth: .infinity)
            Button("Search") { Task { await model.search(model.query) } }
                .disabled(model.loading)
            Text("Format").font(.callout).foregroundStyle(C.dim)
            Picker("Format", selection: $model.format) {
                ForEach(CatalogFormat.allCases) { Text($0.title).tag($0) }
            }
            .labelsHidden()
            .frame(width: 72)
            .onChange(of: model.format) { model.filtersChanged() }
            Text("Scope").font(.callout).foregroundStyle(C.dim)
            Picker("Scope", selection: $model.scope) {
                ForEach(CatalogScope.allCases) { Text($0.title).tag($0) }
            }
            .labelsHidden()
            .frame(width: 138)
            .disabled(model.format == .gguf)
            .onChange(of: model.scope) { model.filtersChanged() }
            Text("Sort").font(.callout).foregroundStyle(C.dim)
            Picker("Sort", selection: $model.sortMode) {
                ForEach(CatalogSort.allCases) { Text($0.title).tag($0) }
            }
            .labelsHidden()
            .frame(width: 140)
            .onChange(of: model.sortMode) { model.filtersChanged() }
        }
    }

    private var downloadedBlock: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .firstTextBaseline) {
                Text("Downloaded")
                    .font(.headline)
                Text(model.query.isEmpty
                     ? "\(visibleDownloaded.count) models · \(humanBytes(visibleDownloaded.reduce(0) { $0 + $1.sizeBytes }))"
                     : "\(visibleDownloaded.count) of \(model.downloaded.count) models")
                    .font(mono(11))
                    .foregroundStyle(C.dim)
            }
            ModelTableHeader(showActions: true)
            ScrollView {
                LazyVStack(spacing: 0) {
                    ForEach(visibleDownloaded) { item in
                        DownloadedModelRow(item: item, model: model, panel: panel,
                                           selected: item.id == model.selected,
                                           onSelect: { model.selected = item.id },
                                           onDelete: { model.pendingDelete = item })
                        Divider().opacity(0.25)
                    }
                    if visibleDownloaded.isEmpty {
                        VStack(alignment: .leading, spacing: 6) {
                            Text(model.query.isEmpty ? "No downloaded models" : "No downloaded models match the search")
                                .font(.callout)
                                .foregroundStyle(C.fg)
                            Text(model.query.isEmpty
                                 ? "Choose a model below and download it to get started."
                                 : "Try a different search or clear it to see all downloaded models.")
                                .font(.callout)
                                .foregroundStyle(C.dim)
                            if !model.query.isEmpty {
                                Button("Clear search") {
                                    model.query = ""
                                    Task { await model.search("") }
                                }
                            }
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.vertical, 14)
                    }
                }
            }
            .frame(height: visibleDownloaded.isEmpty ? 104 : model.downloadedHeight)
        }
    }

    private func listSplitter(totalHeight: CGFloat) -> some View {
        ZStack {
            Rectangle().fill(Color.white.opacity(0.05))
            Image(systemName: "line.3.horizontal")
                .font(.system(size: 11))
                .foregroundStyle(C.dim)
        }
        .frame(height: 14)
        .contentShape(Rectangle())
        .onHover { hovering in
            if hovering { NSCursor.resizeUpDown.push() } else { NSCursor.pop() }
        }
        .gesture(DragGesture(minimumDistance: 1)
            .onChanged { value in
                model.resizeSplit(value.translation.height, totalHeight: totalHeight)
            }
            .onEnded { _ in model.endSplitDrag() })
        .accessibilityLabel("Resize model lists")
        .accessibilityHint("Drag vertically or use Up and Down to resize Downloaded and Hugging Face")
        .accessibilityValue("Downloaded height \(Int(model.downloadedHeight)) points")
        .accessibilityAdjustableAction { direction in
            let delta: CGFloat = direction == .increment ? 24 : -24
            model.nudgeSplit(delta, totalHeight: totalHeight)
        }
        .allowsHitTesting(!visibleDownloaded.isEmpty)
    }

    private var catalogBlock: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .firstTextBaseline) {
                Text("Hugging Face")
                    .font(.headline)
                Text(model.hasMore ? "\(model.models.count) loaded" : "\(model.models.count) models")
                    .font(mono(11))
                    .foregroundStyle(C.dim)
            }
            ModelTableHeader(showActions: true)
            ScrollView {
                LazyVStack(spacing: 0) {
                    ForEach(model.models) { item in
                        CatalogModelRow(item: item, model: model, selected: item.id == model.selected)
                            .onTapGesture { model.selected = item.id }
                        Divider().opacity(0.25)
                    }
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .overlay {
                if !panel.everConnected {
                    ProgressView("Starting the built-in backend…")
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                        .background(Color.black.opacity(0.45))
                } else if model.loading {
                    ProgressView("Ranking models for this Mac…")
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                        .background(Color.black.opacity(0.45))
                } else if model.models.isEmpty {
                    VStack(spacing: 8) {
                        Text("No models match these filters")
                            .font(.callout)
                            .foregroundStyle(C.fg)
                        Text("Try another search or reset the filters.")
                            .font(.callout)
                            .foregroundStyle(C.dim)
                        Button("Reset filters") { Task { await model.resetFilters() } }
                    }
                }
            }
            if model.hasMore {
                Button(model.loadingMore ? "Loading…" : "Load more") {
                    Task { await model.loadMore() }
                }
                .disabled(model.loadingMore)
                .frame(maxWidth: .infinity)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var ratingBreakdown: some View {
        Group {
            if let item = selectedModel {
                if let rating = item.rating {
                    VStack(alignment: .leading, spacing: 4) {
                        Text(item.id)
                            .font(mono(11, .semibold))
                            .lineLimit(1)
                        Text(String(format: "llmfit %.1f · quality %.0f · speed %.0f (%.0f tok/s) · fit %.0f · context %.0f · %.1f GB of %d GB · %@",
                                    rating.score, rating.quality, rating.speed, rating.tps,
                                    rating.fit, rating.context, rating.mem, model.ramGb,
                                    rating.useCase.replacingOccurrences(of: "_", with: " ")))
                            .font(mono(11))
                            .foregroundStyle(C.dim)
                        if let active = item.activeParametersB {
                            Text(String(format: "Parameters %.1fB total · %.1fB active%@",
                                        item.parametersB ?? rating.params, active,
                                        item.tools ? " · tool use" : ""))
                                .font(mono(11))
                                .foregroundStyle(C.dim)
                        }
                    }
                } else {
                    Text("\(item.id) — no llmfit rating; local hardware-fit score \(String(format: "%.1f", item.score))")
                        .font(mono(11))
                        .foregroundStyle(C.dim)
                        .lineLimit(2)
                }
            } else {
                Text("Select a model for its llmfit rating and parameters")
                    .font(mono(11))
                    .foregroundStyle(C.dim)
            }
        }
        .frame(maxWidth: .infinity, minHeight: 42, alignment: .leading)
    }
}

private struct ErrorBanner: View {
    let message: String
    var onDismiss: (() -> Void)? = nil

    var body: some View {
        HStack(alignment: .top, spacing: 9) {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(C.red)
                .accessibilityHidden(true)
            Text(message)
                .font(.callout)
                .foregroundStyle(C.fg)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
            if let onDismiss {
                Button(action: onDismiss) {
                    Image(systemName: "xmark")
                }
                .buttonStyle(.plain)
                .foregroundStyle(C.dim)
                .accessibilityLabel("Dismiss error")
            }
        }
        .padding(10)
        .background(C.red.opacity(0.10), in: RoundedRectangle(cornerRadius: 7))
    }
}

private struct LaunchHelpIcon: View {
    let text: String

    var body: some View {
        Image(systemName: "questionmark.circle")
            .font(.system(size: 12))
            .foregroundStyle(C.dim)
            .help(text)
            .accessibilityLabel(text)
    }
}

private struct MemoryMapView: View {
    let totalGB: Double
    let weightsGB: Double
    let cacheGB: Double
    let contextGB: Double
    let usedGB: Double

    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            GeometryReader { geometry in
                let widths = MemoryMapLayout(totalGB: totalGB, weightsGB: weightsGB,
                                             cacheGB: cacheGB, contextGB: contextGB,
                                             nonLLMGB: usedGB).widths(totalWidth: geometry.size.width)
                HStack(spacing: 0) {
                    Rectangle().fill(C.green).frame(width: widths.weights)
                    Rectangle().fill(C.cyan).frame(width: widths.cache)
                    Rectangle().fill(C.yellow).frame(width: widths.context)
                    Rectangle().fill(C.barBg).frame(width: widths.free)
                    Rectangle().fill(C.red).frame(width: widths.nonLLM)
                }
                .frame(height: 14)
                .clipShape(RoundedRectangle(cornerRadius: 4))
            }
            .frame(height: 14)
            HStack(spacing: 10) {
                HStack(spacing: 4) { Circle().fill(C.green).frame(width: 6, height: 6); Text("Weights") }
                HStack(spacing: 4) { Circle().fill(C.cyan).frame(width: 6, height: 6); Text("Cache") }
                HStack(spacing: 4) { Circle().fill(C.yellow).frame(width: 6, height: 6); Text("Context") }
                HStack(spacing: 4) { Circle().fill(C.dim).frame(width: 6, height: 6); Text("Free / other") }
                HStack(spacing: 4) { Circle().fill(C.red).frame(width: 6, height: 6); Text("Non-LLM") }
            }
            .font(.system(size: 10))
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Memory budget")
        .accessibilityValue(String(format: "Weights %.1f GB, cache %.1f GB, context %.1f GB, other apps %.1f GB", weightsGB, cacheGB, contextGB, usedGB))
    }
}

private struct LaunchGroupBoxStyle: GroupBoxStyle {
    func makeBody(configuration: Configuration) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            configuration.label.font(.headline)
            configuration.content
        }
        .padding(12)
        .background(Color.white.opacity(0.04), in: RoundedRectangle(cornerRadius: 8))
    }
}

private struct LaunchParametersView: View {
    let item: ModelInfo
    @ObservedObject var model: ModelManagerModel
    @ObservedObject var panel: PanelModel

    private func label(_ title: String, _ help: String) -> some View {
        HStack(spacing: 4) {
            Text(title)
            LaunchHelpIcon(text: help)
        }
        .frame(width: 105, alignment: .leading)
    }

    private func number(_ keyPath: WritableKeyPath<LaunchParameters, Int>) -> Binding<Double> {
        Binding(get: { Double(model.launchDraft[keyPath: keyPath]) },
                set: { model.launchDraft[keyPath: keyPath] = Int($0) })
    }

    private var contextTokens: Binding<Double> {
        Binding(get: { Double(model.launchDraft.maxCtx) },
                set: {
                    let context = model.launchMemoryBudget.clampRequestedContext(
                        Int($0), bits: model.launchDraft.kvBits)
                    model.launchDraft.maxCtx = context
                    model.launchDraft.promptCacheBytes = Int(model.launchMemoryBudget.clampCache(
                        Int64(model.launchDraft.promptCacheBytes), contextTokens: context,
                        bits: model.launchDraft.kvBits))
                })
    }

    private func decimal(_ keyPath: WritableKeyPath<LaunchParameters, Double>) -> Binding<Double> {
        Binding(get: { model.launchDraft[keyPath: keyPath] },
                set: { model.launchDraft[keyPath: keyPath] = $0 })
    }

    private var cacheGB: Binding<Double> {
        Binding(get: {
            min(Double(model.launchDraft.promptCacheBytes) / 1024 / 1024 / 1024,
                model.maxCacheGB)
        },
                set: { model.launchDraft.promptCacheBytes = Int(model.launchMemoryBudget.clampCache(
            Int64($0 * 1024 * 1024 * 1024), contextTokens: model.launchDraft.maxCtx,
            bits: model.launchDraft.kvBits)) })
    }

    private var kvGB: Double {
        model.launchMemoryBudget.contextBytes(tokens: model.launchDraft.maxCtx,
                                              bits: model.launchDraft.kvBits) / 1024 / 1024 / 1024
    }

    private var remainingGB: Double { cacheRemainingGB }
    private var activeTotalGB: Double {
        Double(model.launchWeightsBytes) / 1024 / 1024 / 1024 + kvGB
    }
    private var cachePeakGB: Double {
        let allocatedWeights = model.launchIsLoaded ? 0 : model.launchWeightsBytes
        return (Double(allocatedWeights) + Double(model.launchDraft.promptCacheBytes)) / 1024 / 1024 / 1024 + kvGB
    }
    private var cacheRemainingGB: Double {
        let allocatedWeights = model.launchIsLoaded ? 0 : model.launchWeightsBytes
        return (Double(model.launchAvailableBytes) - Double(allocatedWeights)
               - Double(model.launchDraft.promptCacheBytes)) / 1024 / 1024 / 1024 - kvGB
    }
    private var currentRemainingGB: Double { cacheRemainingGB }
    private var sheetWidth: CGFloat { min(640, (NSScreen.main?.visibleFrame.width ?? 800) - 60) }
    private var sheetHeight: CGFloat { min(700, (NSScreen.main?.visibleFrame.height ?? 760) - 80) }
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                Text("Launch settings").font(.title2.weight(.semibold))
                Text("Review memory and generation settings for \(item.id).")
                    .font(.callout)
                    .foregroundStyle(C.dim)
                    .textSelection(.enabled)
            GroupBox("Memory") {
                VStack(spacing: 10) {
                    HStack(spacing: 10) {
                        label("Context", "Maximum tokens allowed for the prompt plus generated response. Larger values use more KV memory.")
                        Slider(value: contextTokens,
                               in: 4096...Double(max(8192, model.maxLaunchContext)), step: 4096)
                        .disabled(model.maxLaunchContext <= 4096)
                        .accessibilityLabel("Context window")
                        .accessibilityValue(tokens(model.launchDraft.maxCtx))
                        .accessibilityHint("Maximum tokens for the prompt and response")
                        Text(tokens(model.launchDraft.maxCtx))
                            .font(mono(11)).frame(width: 50, alignment: .trailing)
                    }
                    HStack(spacing: 10) {
                        label("Prompt cache", "Maximum memory reserved for cached prompts. More cache avoids repeated prefill work but leaves less RAM for context.")
                        Slider(value: cacheGB,
                               in: 0...max(0.01, model.maxCacheGB),
                               step: min(1, max(0.01, model.maxCacheGB)))
                        .disabled(model.maxCacheGB <= 0)
                        .accessibilityLabel("Prompt cache")
                        .accessibilityValue(String(format: "%.0f gigabytes", cacheGB.wrappedValue))
                        Text(String(format: "%.0f GB", cacheGB.wrappedValue))
                            .font(mono(11)).frame(width: 58, alignment: .trailing)
                    }
                    Text(String(format: "Active now: %.1f GB · Full-cache peak: %.1f GB",
                                activeTotalGB, cachePeakGB))
                        .font(mono(11, .semibold))
                        .foregroundStyle(C.fg)
                    Text(String(format: "Remaining after full cache: %.1f GB", cacheRemainingGB))
                        .font(mono(11))
                        .foregroundStyle(cacheRemainingGB < 0 ? C.red : C.cyan)
                    if currentRemainingGB < 0 {
                        Text(String(format: "Over RAM budget by %.1f GB", -currentRemainingGB))
                            .font(mono(11, .semibold))
                            .foregroundStyle(C.red)
                    }
                    MemoryMapView(
                        totalGB: Double(model.ramGb),
                        weightsGB: Double(model.launchWeightsBytes) / 1024 / 1024 / 1024,
                        cacheGB: cacheGB.wrappedValue,
                        contextGB: kvGB,
                        usedGB: Double(panel.derived?.othersBytes ?? 0) / 1024 / 1024 / 1024)
                    Text("Effective context: \(tokens(model.effectiveLaunchContext))"
                         + (model.launchKVBytesPerToken.map {
                             " · \(humanBytes(Int64($0)))/token"
                         } ?? ""))
                        .font(mono(11))
                        .foregroundStyle(C.cyan)
                    Text("Budget \(humanBytes(model.launchAvailableBytes))"
                         + " · Reclaimable \(humanBytes(model.launchReclaimableBytes))"
                         + " · Weights \(humanBytes(model.launchWeightsBytes))"
                         + String(format: " · KV %.1f GB", kvGB)
                         + String(format: " · Remaining %.1f GB", remainingGB))
                        .font(mono(10))
                        .foregroundStyle(remainingGB < 0 ? C.red : C.dim)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .padding(.vertical, 6)
            }
            GroupBox("KV cache") {
                HStack(spacing: 18) {
                    HStack {
                        Picker("Bits", selection: $model.launchDraft.kvBits) {
                            Text("Off").tag(0)
                            Text("4").tag(4)
                            Text("8").tag(8)
                        }
                        LaunchHelpIcon(text: "KV-cache quantization bits. Lower values use less memory but may reduce quality or change server behavior.")
                    }
                    HStack {
                        Picker("Group size", selection: $model.launchDraft.kvGroupSize) {
                            ForEach([16, 32, 64, 128, 256, 512, 1024], id: \.self) {
                                Text("\($0)").tag($0)
                            }
                        }
                        LaunchHelpIcon(text: "Number of values grouped for KV-cache quantization. Grouping changes memory layout and compatibility.")
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.vertical, 6)
            }
            GroupBox("Sampling") {
                VStack(spacing: 10) {
                    HStack(spacing: 10) {
                        label("Temperature", "Randomness of generated text. Higher values are more creative but can become incoherent. Safe maximum is 1.0.")
                        Slider(value: decimal(\LaunchParameters.temperature), in: 0...1, step: 0.05)
                            .accessibilityLabel("Temperature")
                            .accessibilityValue(String(format: "%.2f", model.launchDraft.temperature))
                        Text(String(format: "%.2f", model.launchDraft.temperature))
                            .font(mono(11)).frame(width: 50, alignment: .trailing)
                    }
                    HStack(spacing: 10) {
                        label("Top-p", "Nucleus sampling: only the smallest set of tokens whose cumulative probability reaches this value is considered.")
                        Slider(value: decimal(\LaunchParameters.topP), in: 0.05...1, step: 0.05)
                            .accessibilityLabel("Top-p")
                            .accessibilityValue(String(format: "%.2f", model.launchDraft.topP))
                        Text(String(format: "%.2f", model.launchDraft.topP))
                            .font(mono(11)).frame(width: 50, alignment: .trailing)
                    }
                    HStack(spacing: 10) {
                        label("Top-k", "Limits each step to the K most likely tokens. 0 disables the limit.")
                        Slider(value: number(\LaunchParameters.topK), in: 0...256, step: 1)
                            .accessibilityLabel("Top-k")
                            .accessibilityValue("\(model.launchDraft.topK)")
                        Text("\(model.launchDraft.topK)")
                            .font(mono(11)).frame(width: 50, alignment: .trailing)
                    }
                    HStack(spacing: 10) {
                        label("Min-p", "Filters out tokens whose probability is below this value. 0 disables the filter.")
                        Slider(value: decimal(\LaunchParameters.minP), in: 0...1, step: 0.01)
                            .accessibilityLabel("Min-p")
                            .accessibilityValue(String(format: "%.2f", model.launchDraft.minP))
                        Text(String(format: "%.2f", model.launchDraft.minP))
                            .font(mono(11)).frame(width: 50, alignment: .trailing)
                    }
                    HStack(spacing: 10) {
                        label("Max tokens", "Maximum tokens the model may generate in one response. This does not change the context window; lower values reduce response length and latency.")
                        Slider(value: number(\LaunchParameters.maxTokens),
                               in: 256...Double(max(512, model.effectiveLaunchContext)), step: 256)
                        .disabled(model.effectiveLaunchContext <= 256)
                        .accessibilityLabel("Maximum generated tokens")
                        .accessibilityValue(tokens(model.launchDraft.maxTokens))
                        Text(tokens(model.launchDraft.maxTokens))
                            .font(mono(11)).frame(width: 50, alignment: .trailing)
                    }
                }
                .padding(.vertical, 6)
            }
            GroupBox("Timeouts") {
                VStack(spacing: 10) {
                    HStack(spacing: 10) {
                        label("Load", "Maximum time to wait for model weights to finish loading and the server to become ready.")
                        Slider(value: number(\LaunchParameters.loadTimeout),
                               in: 60...3600, step: 60)
                            .accessibilityLabel("Model load timeout")
                            .accessibilityValue("\(model.launchDraft.loadTimeout / 60) minutes")
                        Text("\(model.launchDraft.loadTimeout / 60)m")
                            .font(mono(11)).frame(width: 50, alignment: .trailing)
                    }
                    HStack(spacing: 10) {
                        label("Client", "Maximum time a client request or streamed response may run before it is cancelled.")
                        Slider(value: number(\LaunchParameters.clientTimeout),
                               in: 60...7200, step: 60)
                            .accessibilityLabel("Client request timeout")
                            .accessibilityValue("\(model.launchDraft.clientTimeout / 60) minutes")
                        Text("\(model.launchDraft.clientTimeout / 60)m")
                            .font(mono(11)).frame(width: 50, alignment: .trailing)
                    }
                }
                .padding(.vertical, 6)
            }
            GroupBox("Agent configuration") {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Loading updates MLX Stack provider entries and harness symlinks. Review every affected path before approving.")
                        .font(.callout)
                        .foregroundStyle(C.dim)
                    ForEach(model.launchConfigPreview.files, id: \.self) { path in
                        Text(path)
                            .font(mono(9))
                            .lineLimit(1)
                            .truncationMode(.middle)
                            .textSelection(.enabled)
                    }
                    Text("Backups: \(model.launchConfigPreview.backupRoot)")
                        .font(mono(9))
                        .foregroundStyle(C.dim)
                        .textSelection(.enabled)
                    Toggle("Back up and apply these agent configuration changes",
                           isOn: $model.approveConfigWrites)
                        .accessibilityLabel("Approve agent configuration backup and changes")
                }
                .padding(.vertical, 6)
            }
            if let error = model.launchError {
                ErrorBanner(message: "Couldn't load saved settings. \(error)")
            }
            if model.loadingLaunch {
                ProgressView("Loading saved settings…")
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            HStack {
                Button("Defaults") { model.resetLaunchSettings() }
                    .disabled(!model.launchSettingsLoaded || model.savingLaunch)
                    .help("Restore recommended values")
                    .accessibilityHint("Restore recommended values")
                Spacer()
                if model.launchError != nil {
                    Button("Retry") { Task { await model.loadLaunchSettings(item) } }
                        .disabled(model.loadingLaunch)
                }
                if model.savingLaunch { ProgressView().controlSize(.small) }
                Button("Cancel") { model.pendingLaunch = nil }
                    .keyboardShortcut(.cancelAction)
                Button("Load model") { Task { await model.saveAndLoad(item, panel: panel) } }
                    .buttonStyle(.borderedProminent)
                    .keyboardShortcut(.defaultAction)
                    .disabled(model.savingLaunch || model.loadingLaunch
                              || !model.launchSettingsLoaded || !model.approveConfigWrites)
            }
            }
        }
        .scrollContentBackground(.hidden)
        .padding(20)
        .frame(width: sheetWidth, height: sheetHeight)
        .background(Color(red: 0.055, green: 0.055, blue: 0.06))
        .environment(\.colorScheme, .dark)
        .groupBoxStyle(LaunchGroupBoxStyle())
        .task { await model.loadLaunchSettings(item) }
    }
}

private let modelActionsWidth: CGFloat = 136

private struct ModelTableHeader: View {
    let showActions: Bool

    var body: some View {
        HStack(spacing: 12) {
            Text("Score").frame(width: 44, alignment: .leading)
            Text("Model").frame(maxWidth: .infinity, alignment: .leading)
            Text("Est. tok/s").frame(width: 70, alignment: .trailing)
            Text("Size").frame(width: 58, alignment: .trailing)
            Text("RAM").frame(width: 68, alignment: .trailing)
            if showActions {
                Text("Actions").frame(width: modelActionsWidth, alignment: .trailing)
            }
        }
        .font(mono(10, .semibold))
        .foregroundStyle(C.dim)
        .padding(.horizontal, 6)
    }
}

private struct ModelDataColumns: View {
    let item: ModelInfo
    let loaded: Bool
    @ObservedObject var model: ModelManagerModel

    private var metadata: String {
        var values: [String] = []
        if let params = item.parametersB { values.append(paramsLabel(params)) }
        if let context = item.context { values.append("\(context) context") }
        if model.hasLoadedCatalog {
            values.append("\(compactCount(item.downloads)) downloads")
            values.append("\(compactCount(item.likes)) likes")
        }
        return values.joined(separator: " · ")
    }

    var body: some View {
        HStack(spacing: 12) {
            Text(item.rating == nil ? "—" : String(format: "%.0f", item.score))
                .foregroundStyle(item.rating == nil ? C.dim : C.cyan)
                .frame(width: 44, alignment: .leading)
                .accessibilityLabel(item.rating == nil ? "No hardware fit rating"
                                    : "Hardware fit score \(Int(item.score)) out of 100")
            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 6) {
                    Link(destination: item.hfURL) {
                        HStack(spacing: 4) {
                            Text(item.id)
                                .underline(model.hoveredModelID == item.id)
                            Image(systemName: "globe")
                                .font(.system(size: 9))
                        }
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(item.fits == false ? C.red : C.fg)
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .onHover { hovering in
                        model.hoveredModelID = hovering ? item.id : nil
                    }
                    .help("Open \(item.id) on Hugging Face")
                    .accessibilityLabel("Open \(item.id) on Hugging Face")
                    if item.helper { modelBadge("helper") }
                    if item.recommended { modelBadge("recommended") }
                    if item.downloading { modelBadge("downloading", color: C.yellow) }
                    if item.gguf { modelBadge("GGUF") }
                    if item.tools { modelBadge("tools") }
                    if loaded { modelBadge("loaded", color: C.green) }
                    if item.deletedBefore { modelBadge("deleted", color: C.yellow) }
                }
                Text(metadata)
                    .font(.system(size: 10))
                    .foregroundStyle(C.dim)
                    .lineLimit(1)
                    .help(metadata)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            Text(item.rating.map { String(format: "%.1f", $0.tps) } ?? "—")
                .frame(width: 70, alignment: .trailing)
                .accessibilityLabel(item.rating.map {
                    "Estimated \($0.tps) tokens per second"
                } ?? "Token speed unavailable")
            VStack(alignment: .trailing, spacing: 2) {
                Text(item.size.isEmpty ? "—" : item.size)
                if item.installed {
                    Text("on disk")
                        .font(.system(size: 9))
                        .foregroundStyle(C.dim)
                }
            }
            .frame(width: 58, alignment: .trailing)
            VStack(alignment: .trailing, spacing: 2) {
                Text(ramLabel(item))
                    .foregroundStyle(item.fits == false ? C.red : C.fg)
                if item.fits == false {
                    Text("too large")
                        .font(.system(size: 9, weight: .medium))
                        .foregroundStyle(C.red)
                }
            }
            .frame(width: 68, alignment: .trailing)
            .accessibilityElement(children: .combine)
        }
        .font(mono(11))
    }
}

private struct DownloadProgressView: View {
    let item: ModelInfo

    private var fraction: Double? {
        guard item.downloadTotal > 0 else { return nil }
        return min(1, Double(item.downloadBytes) / Double(item.downloadTotal))
    }

    private var label: String {
        let speed = item.downloadRate > 1024
            ? " · \(humanBytes(Int64(item.downloadRate)))/s"
            : ""
        let prefix = item.downloadPaused ? "Paused · " : ""
        guard let fraction else {
            return prefix + "\(humanBytes(item.downloadBytes)) downloaded\(speed)"
        }
        return prefix + String(format: "%.0f%%", fraction * 100)
            + " · \(humanBytes(item.downloadBytes))/\(humanBytes(item.downloadTotal))"
            + speed
    }

    var body: some View {
        HStack(spacing: 8) {
            if let fraction {
                ProgressView(value: fraction)
            } else {
                ProgressView().controlSize(.small)
            }
            Text(label)
                .font(mono(10))
                .foregroundStyle(C.dim)
                .lineLimit(1)
        }
        .padding(.leading, 62)
        .frame(maxWidth: .infinity, alignment: .leading)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("\(item.id) download")
        .accessibilityValue(label)
    }
}

private struct DownloadedModelRow: View {
    let item: ModelInfo
    @ObservedObject var model: ModelManagerModel
    @ObservedObject var panel: PanelModel
    let selected: Bool
    let onSelect: () -> Void
    let onDelete: () -> Void

    private var loaded: Bool {
        (panel.mainRunning && panel.derived?.model == item.id)
            || panel.derived?.smallModel == item.id
    }
    private var active: Bool { loaded }
    private var loading: Bool {
        panel.loadingModel == item.id || panel.launch?.model == item.id
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 10) {
                ModelDataColumns(item: item, loaded: loaded, model: model)
                HStack(spacing: 6) {
                    if item.downloading {
                        Button(item.downloadPaused ? "Resume" : "Pause") {
                            Task { await model.pauseDownload(item.id, resume: item.downloadPaused) }
                        }
                        Button("Cancel", role: .destructive) {
                            model.cancelDownload(item)
                        }
                        .accessibilityLabel("Cancel download of \(item.id)")
                    } else {
                        if model.deleting == item.id {
                            ProgressView().controlSize(.small)
                                .accessibilityLabel("Deleting \(item.id)")
                        }
                        if loading {
                            ProgressView().controlSize(.small)
                                .accessibilityLabel("Loading \(item.id)")
                        } else {
                            Button(active ? "Unload" : "Load…") {
                                if active {
                                    panel.unloadModel(item.id)
                                } else {
                                    model.beginLaunch(item)
                                }
                            }
                            .controlSize(.small)
                            .disabled(!panel.canSwitch || panel.modelActionInProgress || model.deleting != nil)
                            .help(active ? "Stop this model" : "Review settings before loading \(item.id)")
                            .accessibilityLabel(active ? "Unload \(item.id)" : "Load \(item.id)")
                        }
                        Button(role: .destructive, action: onDelete) {
                            Label("Delete", systemImage: "trash")
                                .labelStyle(.iconOnly)
                        }
                        .controlSize(.small)
                        .disabled(active || loading || model.deleting != nil)
                        .help("Delete downloaded files")
                        .accessibilityLabel("Delete \(item.id) from disk")
                    }
                }
                .controlSize(.small)
                .frame(width: modelActionsWidth, alignment: .trailing)
            }
            if item.downloading {
                DownloadProgressView(item: item)
            }
            if loading, let progress = panel.launch {
                LaunchProgressView(progress: progress)
                    .padding(.leading, 62)
            }
            if let error = panel.loadError, error.model == item.id {
                Text("Couldn't launch \(item.id): \(error.reason ?? "exit status \(error.status)"). Review the log lines below before retrying.")
                    .font(.callout)
                    .foregroundStyle(C.red)
                LogRowsView(rows: panel.derived?.logRows ?? [], followsTail: false)
                    .frame(minHeight: 100, maxHeight: 240)
            }
        }
        .padding(.vertical, 7)
        .padding(.horizontal, 6)
        .background(loaded ? C.green.opacity(0.14)
                    : selected ? Color.white.opacity(0.08) : .clear)
        .contentShape(Rectangle())
        .onTapGesture(perform: onSelect)
        .accessibilityAction(named: "Show model details", onSelect)
    }
}

private struct CatalogModelRow: View {
    let item: ModelInfo
    @ObservedObject var model: ModelManagerModel
    let selected: Bool

    private var busy: Bool {
        item.downloading || model.startingDownloads.contains(item.id)
    }
    private var buttonTitle: String {
        if item.downloadError != nil { return "Retry download" }
        if busy { return "Downloading…" }
        return item.deletedBefore ? "Download again" : "Download"
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 10) {
                ModelDataColumns(item: item, loaded: false, model: model)
                if item.downloading {
                    HStack(spacing: 6) {
                        Button(item.downloadPaused ? "Resume" : "Pause") {
                            Task { await model.pauseDownload(item.id, resume: item.downloadPaused) }
                        }
                        Button("Cancel", role: .destructive) {
                            model.cancelDownload(item)
                        }
                        .accessibilityLabel("Cancel download of \(item.id)")
                    }
                    .controlSize(.small)
                    .frame(width: modelActionsWidth, alignment: .trailing)
                } else {
                    Button(buttonTitle) {
                        Task { await model.download(item) }
                    }
                    .controlSize(.small)
                    .disabled(busy)
                    .frame(width: modelActionsWidth, alignment: .trailing)
                    .accessibilityLabel("\(buttonTitle) \(item.id)")
                }
            }
            if item.downloading {
                DownloadProgressView(item: item)
            }
            if let error = item.downloadError {
                Text(error)
                    .font(.callout)
                    .foregroundStyle(C.red)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .padding(.vertical, 8)
        .padding(.horizontal, 6)
        .background(selected ? Color.white.opacity(0.08) : .clear)
        .contentShape(Rectangle())
        .accessibilityAction(named: "Show model details") { model.selected = item.id }
    }
}

private func modelBadge(_ text: String, color: Color = C.dim) -> some View {
    Text(text)
        .font(.system(size: 9, weight: .medium))
        .foregroundStyle(color)
        .padding(.horizontal, 4)
        .padding(.vertical, 1)
        .overlay(RoundedRectangle(cornerRadius: 3).stroke(color.opacity(0.55)))
}

private func paramsLabel(_ value: Double) -> String {
    String(format: "%.1fB", value)
}

private func ramLabel(_ item: ModelInfo) -> String {
    item.ramGb.map { String(format: "~%.0f GB", $0) } ?? "? GB"
}

private func compactCount(_ value: Int) -> String {
    if value >= 1_000_000 { return String(format: "%.1fM", Double(value) / 1_000_000) }
    if value >= 1_000 { return String(format: "%.0fK", Double(value) / 1_000) }
    return String(value)
}
