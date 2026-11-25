// MLXStack.swift — the native app: a status-bar item and one window with the
// Models, Monitor and Tools sections.
//
// Built by ./build.sh into build/MLX Stack.app. The app owns the Python
// backend that ships inside the bundle (Backend/) and needs no terminal: it
// starts the local API (python -m local_llm.widget), which starts the stack
// — client proxy, main model and helper model — when a model is loaded.
//
// Monitor polls GET http://127.0.0.1:<port>/snapshot once a second, keeps
// 240-sample histories and decode-only tok/s logic, and renders: header
// chips, the summary, the Generation and Context usage panels, CPU + cores,
// memory, processes, logs (errors red). macOS 14+.

import AppKit
import Foundation
import SwiftUI

// MARK: - entry

@main
@MainActor
struct MLXStackApp {
    private static var retainedDelegate: AppDelegate?

    static func main() {
        let app = NSApplication.shared
        let delegate = AppDelegate()
        retainedDelegate = delegate   // NSApplication.delegate is non-retaining
        app.delegate = delegate
        app.setActivationPolicy(.regular)   // Dock and ⌘Tab presence; no hiding on focus loss
        app.appearance = NSAppearance(named: .darkAqua)
        app.run()
    }
}

/// "--port <n>" (or "--port=<n>"); default WIDGET_PORT of the launcher.
enum PortArg {
    static let value: Int = {
        let args = CommandLine.arguments
        var i = 0
        while i < args.count {
            if args[i] == "--port", i + 1 < args.count, let p = Int(args[i + 1]) {
                return p
            }
            if args[i].hasPrefix("--port="), let p = Int(args[i].dropFirst("--port=".count)) {
                return p
            }
            i += 1
        }
        return 19798
    }()
}

@MainActor
final class AppNavigation: ObservableObject {
    enum Screen { case models, monitor, tools }
    @Published var screen: Screen = .models
    @Published private(set) var hasChosenScreen = false

    func show(_ screen: Screen) {
        self.screen = screen
        hasChosenScreen = true
    }
}

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate, NSMenuDelegate {
    let model = PanelModel(port: PortArg.value)
    var window: NSWindow?
    private var statusItem: NSStatusItem?
    private var apiProcess: Process?
    private let navigation = AppNavigation()
    private lazy var modelManager = ModelManagerModel(port: model.port)

    func applicationWillFinishLaunching(_ notification: Notification) {
        let menu = NSMenu()
        let appItem = NSMenuItem()
        let appMenu = NSMenu(title: "MLX Stack")
        appMenu.addItem(withTitle: "About MLX Stack",
                        action: #selector(NSApplication.orderFrontStandardAboutPanel(_:)),
                        keyEquivalent: "")
        appMenu.addItem(.separator())
        appMenu.addItem(withTitle: "Quit MLX Stack",
                        action: #selector(NSApplication.terminate(_:)),
                        keyEquivalent: "q")
        appItem.submenu = appMenu
        menu.addItem(appItem)
        let editItem = NSMenuItem()
        let editMenu = NSMenu(title: "Edit")
        editMenu.addItem(withTitle: "Copy", action: #selector(NSText.copy(_:)),
                         keyEquivalent: "c")
        let pasteItem = NSMenuItem(title: "Paste",
                                   action: #selector(pasteIntoSearch(_:)),
                                   keyEquivalent: "v")
        pasteItem.target = self
        editMenu.addItem(pasteItem)
        editItem.submenu = editMenu
        menu.addItem(editItem)
        let windowItem = NSMenuItem()
        let windowMenu = NSMenu(title: "Window")
        let models = NSMenuItem(title: "Models", action: #selector(showModelManager),
                                keyEquivalent: "1")
        models.target = self
        models.keyEquivalentModifierMask = [.command]
        windowMenu.addItem(models)
        let monitor = NSMenuItem(title: "Monitor", action: #selector(showMonitor),
                                 keyEquivalent: "2")
        monitor.target = self
        monitor.keyEquivalentModifierMask = [.command]
        windowMenu.addItem(monitor)
        let tools = NSMenuItem(title: "Tools", action: #selector(showTools),
                               keyEquivalent: "3")
        tools.target = self
        tools.keyEquivalentModifierMask = [.command]
        windowMenu.addItem(tools)
        windowMenu.addItem(.separator())
        windowMenu.addItem(withTitle: "Minimize",
                           action: #selector(NSWindow.miniaturize(_:)),
                           keyEquivalent: "m")
        windowItem.submenu = windowMenu
        menu.addItem(windowItem)
        NSApplication.shared.mainMenu = menu
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSLog("MLXStack applicationDidFinishLaunching")
        let host = NSHostingView(rootView: AppRootView(
            panel: model, manager: modelManager, navigation: navigation))
        host.autoresizingMask = [.width, .height]

        let w = NSWindow(
            contentRect: Self.initialWindowFrame(),
            styleMask: [.titled, .closable, .miniaturizable, .resizable],
            backing: .buffered, defer: false)
        w.title = "MLX Stack"
        w.sharingType = .readOnly
        w.isOpaque = true
        w.backgroundColor = .windowBackgroundColor
        w.hasShadow = true
        w.hidesOnDeactivate = false
        w.isMovableByWindowBackground = true
        w.level = .normal
        // Ordinary non-modal window: other apps can cover and activate freely.
        // Joins every (non-fullscreen) Space without floating above them.
        w.collectionBehavior = [.canJoinAllSpaces, .ignoresCycle]
        w.minSize = NSSize(width: 980, height: 520)
        w.contentMinSize = NSSize(width: 980, height: 520)
        w.isReleasedWhenClosed = false
        w.contentView = host
        w.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
        window = w
        startAPI()
        model.start()
        installStatusItem()
    }
    private var stoppingAPI = false

    private func startAPI() {
        guard let launcher = Bundle.main.resourceURL?
            .appendingPathComponent("backend/launch-widget.sh") else {
            NSLog("MLXStack bundled backend is missing")
            return
        }
        NSLog("MLXStack API launcher %@", launcher.path)
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/sh")
        process.arguments = [launcher.path]
        var env = ProcessInfo.processInfo.environment
        env["WIDGET_PORT"] = String(model.port)
        env.removeValue(forKey: "PYTHONHOME")
        env.removeValue(forKey: "PYTHONPATH")
        process.environment = env
        process.terminationHandler = { [weak self] process in
            let status = process.terminationStatus
            NSLog("MLXStack API exited with status %d", status)
            DispatchQueue.main.async {
                guard let self, !self.stoppingAPI else { return }
                self.showBackendFailure("The local model service exited with status \(status).")
            }
        }
        do {
            try process.run()
            apiProcess = process
            NSLog("MLXStack API pid %d", process.processIdentifier)
        } catch {
            NSLog("MLXStack API launch error %@", error.localizedDescription)
            showBackendFailure("The local model service could not start. \(error.localizedDescription)")
        }
    }

    private func showBackendFailure(_ detail: String) {
        let alert = NSAlert()
        alert.messageText = "MLX Stack backend stopped"
        alert.informativeText = detail + " Open the log to inspect the problem, then relaunch MLX Stack."
        alert.alertStyle = .warning
        alert.addButton(withTitle: "Show Log")
        alert.addButton(withTitle: "OK")
        if alert.runModal() == .alertFirstButtonReturn {
            let log = FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent(".mlx-stack/logs/mlx-stack.log")
            NSWorkspace.shared.activateFileViewerSelecting([log])
        }
    }

    func applicationWillTerminate(_ notification: Notification) {
        stoppingAPI = true
        if let apiProcess, apiProcess.isRunning {
            apiProcess.terminate()
        }
    }

    private static func initialWindowFrame() -> NSRect {
        let screen = NSScreen.main?.visibleFrame
            ?? NSRect(x: 0, y: 0, width: 1280, height: 800)
        let width = min(1180, screen.width - 40)
        let height = min(800, screen.height - 40)
        return NSRect(x: screen.midX - width / 2,
                      y: screen.midY - height / 2,
                      width: width, height: height)
    }

    private func presentWindow() {
        guard let window else { return }
        if window.isMiniaturized { window.deminiaturize(nil) }
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    private func installStatusItem() {
        let item = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        guard let button = item.button else { return }
        let image = NSImage(size: NSSize(width: 18, height: 18), flipped: false) { _ in
            // The application icon's M ribbon, reduced to a legible menu-bar mark.
            let mark = NSBezierPath()
            mark.move(to: NSPoint(x: 2.5, y: 3))
            mark.line(to: NSPoint(x: 6.2, y: 14.5))
            mark.line(to: NSPoint(x: 9, y: 3))
            mark.line(to: NSPoint(x: 11.8, y: 14.5))
            mark.line(to: NSPoint(x: 15.5, y: 3))
            mark.lineWidth = 3.2
            mark.lineCapStyle = .round
            mark.lineJoinStyle = .round
            NSColor.black.setStroke()
            mark.stroke()
            return true
        }
        image.isTemplate = true
        button.image = image
        button.toolTip = "MLX Stack — monitor and models"
        button.setAccessibilityTitle("MLX Stack")
        let menu = NSMenu()
        menu.delegate = self
        item.menu = menu
        statusItem = item
    }

    func menuNeedsUpdate(_ menu: NSMenu) {
        menu.removeAllItems()
        let open = NSMenuItem(
            title: window?.isVisible == true ? "Hide MLX Stack" : "Open MLX Stack",
            action: #selector(togglePanel), keyEquivalent: "")
        open.target = self
        menu.addItem(open)
        let manage = NSMenuItem(title: "Manage models",
                                action: #selector(showModelManager), keyEquivalent: "")
        manage.target = self
        menu.addItem(manage)
        menu.addItem(.separator())

        let title = model.canSwitch ? "Load model" : "Models — terminal stack active"
        let header = NSMenuItem(title: title, action: nil, keyEquivalent: "")
        header.isEnabled = false
        menu.addItem(header)
        for name in model.availableModels {
            let item = NSMenuItem(title: name, action: #selector(selectModel(_:)),
                                  keyEquivalent: "")
            item.target = self
            item.representedObject = name
            item.state = model.derived?.model == name ? .on : .off
            item.isEnabled = model.canSwitch && !model.modelActionInProgress
            menu.addItem(item)
        }
        if model.availableModels.isEmpty {
            let empty = NSMenuItem(title: "No downloaded models", action: nil,
                                   keyEquivalent: "")
            empty.isEnabled = false
            menu.addItem(empty)
        }
        menu.addItem(.separator())
        let quit = NSMenuItem(title: "Exit", action: #selector(quit),
                              keyEquivalent: "")
        quit.target = self
        menu.addItem(quit)
    }

    @objc private func pasteIntoSearch(_ sender: Any?) {
        guard let text = NSPasteboard.general.string(forType: .string) else { return }
        modelManager.query = text
    }

    @objc private func showModelManager() {
        navigation.show(.models)
        presentWindow()
    }

    @objc private func showMonitor() {
        navigation.show(.monitor)
        presentWindow()
    }

    @objc private func showTools() {
        navigation.show(.tools)
        presentWindow()
    }

    @objc private func selectModel(_ sender: NSMenuItem) {
        guard let name = sender.representedObject as? String else { return }
        model.runModel(name)
        navigation.screen = .models
    }

    @objc private func quit() {
        NSApp.terminate(nil)
    }

    @objc private func togglePanel() {
        guard let window else { return }
        if window.isMiniaturized {
            presentWindow()
        } else if window.isVisible {
            window.orderOut(nil)
        } else {
            presentWindow()
        }
    }

    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool) -> Bool {
        presentWindow()
        return false
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        false   // the status item owns the app lifetime; Cmd+Q terminates
    }
}

/// Dark, slightly translucent desktop material behind the panel.
struct VisualEffectBackground: NSViewRepresentable {
    func makeNSView(context: Context) -> NSVisualEffectView {
        let v = NSVisualEffectView()
        v.material = .underWindowBackground
        v.blendingMode = .behindWindow
        v.state = .active
        return v
    }

    func updateNSView(_ nsView: NSVisualEffectView, context: Context) {}
}

// MARK: - snapshot payload

struct Snapshot: Decodable {
    struct Port: Decodable {
        let label: String
        let port: Int
        let up: Bool
    }

    struct Ports: Decodable {
        let main: Port
        let small: Port?
        let proxy: Port
    }

    struct CPU: Decodable {
        let total: Double
        let cores: [Double]
    }

    struct Mem: Decodable {
        let used: Int64
        let total: Int64
        let available: Int64
        let percent: Double
    }

    struct Swap: Decodable {
        let used: Int64
        let total: Int64
        let percent: Double
    }

    struct Proc: Decodable {
        let name: String
        let pid: Int
        let cpu: Double
        let rss: Int64
        let rssPercent: Double
        let threads: Int
    }

    struct ProcSummary: Decodable {
        let ours: Int64
        let others: Int64
        let oursPercent: Double
    }

    struct Eater: Decodable {
        let name: String
        let pid: Int
        let rss: Int64
    }

    struct Perf: Decodable {
        let liveSeq: Int
        let liveRate: Double
        let preSeq: Int
        let preRate: Double
        let summary: String
        let context: Int?
        let active: Int?
        let requests: Int
    }

    struct Client: Decodable {
        let name: String
        let command: String
        let configPath: String?
        let apiBase: String?
        let model: String?
        let role: String?
    }

    struct LogLine: Decodable {
        let tag: String
        let text: String
        let error: Bool
    }

    struct LaunchError: Decodable {
        let model: String
        let status: Int
    }

    let model: String
    let smallModel: String?
    let launchError: LaunchError?
    let backend: String
    let uptimeSeconds: Int
    let maxInput: Int
    let configHome: String?
    let availableModels: [String]
    let canSwitch: Bool
    let ports: Ports
    let cpu: CPU
    let memory: Mem
    let swap: Swap?
    let processes: [Proc]
    let processSummary: ProcSummary
    let topEaters: [Eater]
    let main: Perf
    let small: Perf?
    let clients: [Client]
    let logs: [LogLine]
}

// MARK: - local histories (240 samples, four minutes at one per second)

let historySeconds = 240

struct Ring {
    private(set) var values: [Double] = []
    private let cap: Int

    init(capacity: Int = historySeconds) {
        self.cap = capacity
    }

    mutating func append(_ v: Double) {
        values.append(v)
        if values.count > cap {
            values.removeFirst(values.count - cap)
        }
    }
}

/// What one second of a model's life looked like, sampled once per poll.
enum Activity {
    static let idle: Double = 0        // no request in flight: waiting for the client
    static let stalled: Double = 1     // request in flight, no token progress this second
    static let prefill: Double = 2     // prompt being processed
    static let decode: Double = 3      // tokens being generated
}

/// Per-model decode state: which seq numbers were last seen and what they showed.
struct TokSt {
    var liveSeq = -1
    var preSeq = -1
    var idle = 1 << 30
    var cur = 0.0
    var txt = "idle"
    var summary = ""
    var peak = 0.0
}

struct RequestSt {
    var total = 0
    var current = 0
    var initialized = false
    var lastRequest: Date?
    var averageInterval: Double?
}

// MARK: - immutable view model built once per poll

struct Chip {
    let service: String
    let port: Int
    let up: Bool
}

enum MonitorModel: String, CaseIterable, Identifiable {
    case main, small
    var id: String { rawValue }
    var title: String { self == .main ? "Main" : "Small" }
}

struct Row {
    let text: String
    let color: Color
}

struct TokCol {
    let samples: [Double]              // decode tok/s per second, 0 while not generating
    let scale: Double
    let cur: Double
    let txt: String
    let avg: Double?                   // mean over the seconds that were generating
    let peak: Double
    let footer: String
}

struct RequestCol {
    let samples: [Double]              // requests started per second
    let total: Int
    let perMinute: Double              // since the model was loaded
    let lastAgo: Double?
    let averageInterval: Double?
}

/// Shares of the history window the model spent in each activity.
struct ActivityCol {
    let samples: [Double]
    let generating: Double
    let reading: Double
    let waiting: Double
}

struct CtxCol {
    let barFrac: Double
    let barSuffix: String             // "] 4.1K/110K"
    let samples: [Double]
    let scale: Double                 // graph scale, dynamic
    let capacity: Double              // absolute context cap
    let current: Int
    let footer: String                // "TTFT … · uncached … · active …"
}

struct Derived {
    let model: String
    let smallModel: String?
    let uptime: String
    let configHome: String
    let chips: [Chip]
    let coreCount: Int
    let tokMain: TokCol
    let tokSmall: TokCol?
    let requestsMain: RequestCol
    let requestsSmall: RequestCol?
    let activityMain: ActivityCol
    let activitySmall: ActivityCol?
    let activeRequests: Int?
    let activeRequestsSmall: Int?
    let ctx: CtxCol
    let ctxSmall: CtxCol?
    let cpuTotal: Double
    let cpuSuffix: String
    let cpuHist: [Double]
    let cores: [Double]
    let clients: [Snapshot.Client]
    let procRows: [Row]
    let othersBytes: Int64
    let memFrac: Double
    let memSuffix: String
    let memoryDetail: String
    let memLow: Bool
    let swapFrac: Double?
    let swapSuffix: String?
    let topMem: Row?
    let logRows: [Row]
    let errorCount: Int
}

// MARK: - poller + state machine

@MainActor
final class PanelModel: ObservableObject {
    let port: Int

    @Published private(set) var derived: Derived?
    @Published private(set) var offline = true
    @Published private(set) var everConnected = false
    @Published private(set) var availableModels: [String] = []
    @Published private(set) var canSwitch = false
    @Published private(set) var mainRunning = false
    @Published private(set) var modelActionInProgress = false
    @Published private(set) var loadingModel: String?
    @Published private(set) var loadError: Snapshot.LaunchError?
    @Published var actionError: String?
    @Published var copiedCommand: String?
    @Published var selectedModel: MonitorModel = .main
    @Published var showCPUCores = false
    @Published var showProcesses = false
    @Published var showLogs = false

    private var tokMain = TokSt()
    private var ctxSmallCur = 0
    private var ringCtxS = Ring()
    private var tokSmall = TokSt()
    private var ringTok = Ring()
    private var ringTokS = Ring()
    private var ringRequests = Ring()
    private var ringRequestsS = Ring()
    private var ringActivity = Ring()
    private var ringActivityS = Ring()
    private var requestMain = RequestSt()
    private var requestSmall = RequestSt()
    private var ringCtx = Ring()
    private var ringCpu = Ring()
    private var ctxCur = 0
    private var lastModel: String?
    private var lastSmallModel: String?
    private var lastUptime: Int?

    private let session: URLSession
    private var pollTask: Task<Void, Never>?

    private static let doneRx = try! NSRegularExpression(
        pattern: ##"^(\d+) tok.*? prompt (\d+) tok, (\d+) cached(?: \(([\d.]+)s to first\))?"##)

    init(port: Int) {
        self.port = port
        let cfg = URLSessionConfiguration.ephemeral
        cfg.timeoutIntervalForRequest = 2
        cfg.timeoutIntervalForResource = 3
        cfg.requestCachePolicy = .reloadIgnoringLocalCacheData
        session = URLSession(configuration: cfg)
    }

    func start() {
        pollTask?.cancel()
        guard let url = URL(string: "http://127.0.0.1:\(port)/snapshot") else { return }
        pollTask = Task { [weak self] in
            while !Task.isCancelled {
                await self?.poll(url: url)
                try? await Task.sleep(for: .seconds(1))   // sequential: never overlaps
            }
        }
    }

    func runModel(_ name: String) {
        guard canSwitch, !modelActionInProgress,
              let url = URL(string: "http://127.0.0.1:\(port)/run") else { return }
        modelActionInProgress = true
        actionError = nil
        loadError = nil
        Task {
            defer { modelActionInProgress = false }
            do {
                var request = URLRequest(url: url)
                request.httpMethod = "POST"
                request.timeoutInterval = 40
                request.setValue("application/json", forHTTPHeaderField: "Content-Type")
                request.httpBody = try JSONSerialization.data(withJSONObject: ["model": name])
                let (data, response) = try await session.data(for: request)
                let status = (response as? HTTPURLResponse)?.statusCode ?? 0
                if status != 202 {
                    let payload = try? JSONSerialization.jsonObject(with: data) as? [String: String]
                    throw NSError(domain: "MLXStack", code: status,
                                  userInfo: [NSLocalizedDescriptionKey:
                                      payload?["error"] ?? "model launch failed"])
                }
            } catch {
                if loadingModel == name { loadingModel = nil }
                actionError = "Couldn't load \(name). \(error.localizedDescription)"
            }
        }
    }

    func unloadModel(_ name: String) {
        guard !modelActionInProgress,
              let url = URL(string: "http://127.0.0.1:\(port)/stop") else { return }
        modelActionInProgress = true
        actionError = nil
        Task {
            defer { modelActionInProgress = false }
            do {
                var request = URLRequest(url: url)
                request.httpMethod = "POST"
                request.setValue("application/json", forHTTPHeaderField: "Content-Type")
                request.httpBody = try JSONSerialization.data(withJSONObject: ["model": name])
                let (data, response) = try await session.data(for: request)
                let status = (response as? HTTPURLResponse)?.statusCode ?? 0
                if status != 200 {
                    let payload = try? JSONSerialization.jsonObject(with: data) as? [String: String]
                    throw NSError(domain: "MLXStack", code: status,
                                  userInfo: [NSLocalizedDescriptionKey:
                                      payload?["error"] ?? "model unload failed"])
                }
            } catch {
                actionError = "Couldn't unload \(name). \(error.localizedDescription)"
            }
        }
    }

    func copyClient(_ command: String) {
        copyText(command)
        copiedCommand = command
        Task {
            try? await Task.sleep(for: .seconds(2))
            if copiedCommand == command { copiedCommand = nil }
        }
    }

    private func poll(url: URL) async {
        do {
            let (data, _) = try await session.data(from: url)
            let decoder = JSONDecoder()
            decoder.keyDecodingStrategy = .convertFromSnakeCase
            let snap = try decoder.decode(Snapshot.self, from: data)
            apply(snap)
            everConnected = true
            offline = false
        } catch {
            offline = true   // histories are kept; the view shows the offline state
        }
    }

    /// One poll of decode bookkeeping: a changed live_seq shows live_rate
    /// (and feeds the graph), a changed pre_seq shows prefill, anything else
    /// decays to idle after 3 grace ticks. Returns whether this poll saw new
    /// tokens / new prefill progress.
    private func stepTok(_ st: inout TokSt, _ p: Snapshot.Perf, _ ring: inout Ring) -> (live: Bool, pre: Bool) {
        let liveNew = p.liveSeq != st.liveSeq
        let preNew = p.preSeq != st.preSeq
        if liveNew {
            st.idle = 0
            st.cur = p.liveRate
            st.txt = String(format: "%5.1f tok/s", p.liveRate)
            st.peak = max(st.peak, p.liveRate)
        } else if preNew {
            st.idle = 0
            st.cur = p.preRate
            st.txt = String(format: "%4.0f tok/s prefill", p.preRate)
        } else {
            st.idle += 1
            if st.idle > 3 {
                st.cur = 0.0
                st.txt = "idle"
            }
        }
        st.liveSeq = p.liveSeq
        st.preSeq = p.preSeq
        st.summary = p.summary
        // decode-only graph: zeros mark idle/prefill periods
        ring.append(liveNew ? p.liveRate : 0.0)
        return (liveNew, preNew)
    }

    private func stepIdle(_ st: inout TokSt, _ ring: inout Ring) {
        st.idle += 1
        if st.idle > 3 {
            st.cur = 0.0
            st.txt = "idle"
        }
        ring.append(0.0)
    }

    private func stepRequests(_ state: inout RequestSt, _ count: Int, ring: inout Ring) {
        if state.initialized {
            state.current = count >= state.total ? count - state.total : count
            if count > state.total {
                let now = Date()
                if let last = state.lastRequest {
                    let interval = now.timeIntervalSince(last)
                    state.averageInterval = state.averageInterval.map {
                        $0 * 0.8 + interval * 0.2
                    } ?? interval
                }
                state.lastRequest = now
            }
        } else {
            state.initialized = true
            state.current = 0
        }
        state.total = count
        ring.append(Double(state.current))
    }

    private func requestCol(_ state: RequestSt, _ ring: Ring, uptime: Int) -> RequestCol {
        RequestCol(samples: ring.values,
                   total: state.total,
                   perMinute: uptime > 0 ? Double(state.total) / Double(uptime) * 60 : 0,
                   lastAgo: state.lastRequest.map { Date().timeIntervalSince($0) },
                   averageInterval: state.averageInterval)
    }

    private func activityCol(_ ring: Ring) -> ActivityCol {
        let n = ring.values.count
        func share(_ matches: [Double]) -> Double { n > 0 ? Double(matches.count) / Double(n) : 0 }
        return ActivityCol(samples: ring.values,
                           generating: share(ring.values.filter { $0 == Activity.decode }),
                           reading: share(ring.values.filter { $0 == Activity.prefill || $0 == Activity.stalled }),
                           waiting: share(ring.values.filter { $0 == Activity.idle }))
    }

    private func tokCol(_ st: TokSt, _ ring: Ring) -> TokCol {
        let generating = ring.values.filter { $0 > 0 }
        return TokCol(samples: ring.values,
                      // a multiple of 50 keeps the half-scale tick a whole number
                      scale: dynamicScale(ring.values, minimum: 50, step: 50),
                      cur: st.cur,
                      txt: st.txt,
                      avg: generating.isEmpty ? nil : generating.reduce(0, +) / Double(generating.count),
                      peak: st.peak,
                      footer: st.summary.isEmpty
                          ? "no completions yet — tok/s appears during generation"
                          : "last: " + st.summary)
    }

    /// Token progress wins, then a request in flight, else the model is idle.
    private static func activity(live: Bool, pre: Bool, inFlight: Int) -> Double {
        live ? Activity.decode : pre ? Activity.prefill : inFlight > 0 ? Activity.stalled : Activity.idle
    }

    private func apply(_ snap: Snapshot) {
        let restarted = lastUptime.map { snap.uptimeSeconds < $0 } ?? false
        if lastModel != snap.model || lastSmallModel != snap.smallModel || restarted {
            tokMain = TokSt()
            tokSmall = TokSt()
            requestMain = RequestSt()
            requestSmall = RequestSt()
            ringTok = Ring()
            ringTokS = Ring()
            ringRequests = Ring()
            ringRequestsS = Ring()
            ringActivity = Ring()
            ringActivityS = Ring()
            ringCtx = Ring()
            ringCtxS = Ring()
            ctxCur = 0
            ctxSmallCur = 0
        }
        lastModel = snap.model
        lastSmallModel = snap.smallModel
        lastUptime = snap.uptimeSeconds
        availableModels = snap.availableModels
        canSwitch = snap.canSwitch
        mainRunning = snap.ports.main.up
        let mainStep = stepTok(&tokMain, snap.main, &ringTok)
        stepRequests(&requestMain, snap.main.requests, ring: &ringRequests)
        ringActivity.append(Self.activity(live: mainStep.live, pre: mainStep.pre,
                                          inFlight: snap.main.active ?? 0))
        if let sp = snap.small {
            let smallStep = stepTok(&tokSmall, sp, &ringTokS)
            stepRequests(&requestSmall, sp.requests, ring: &ringRequestsS)
            ringActivityS.append(Self.activity(live: smallStep.live, pre: smallStep.pre,
                                               inFlight: sp.active ?? 0))
        } else if snap.ports.small != nil {
            stepIdle(&tokSmall, &ringTokS)
            ringRequestsS.append(0)
            ringActivityS.append(Activity.idle)
        }
        if let c = snap.main.context { ctxCur = c }
        ringCtx.append(Double(ctxCur))
        if let sp = snap.small {
            if let c = sp.context { ctxSmallCur = c }
            ringCtxS.append(Double(ctxSmallCur))
        } else {
            ringCtxS.append(0)
        }
        ringCpu.append(snap.cpu.total)

        let serviceNames = ["mlx": "Main API", "small": "Helper API", "proxy": "Client proxy"]
        let chips: [Chip] = [snap.ports.main, snap.ports.small, snap.ports.proxy]
            .compactMap { $0 }
            .map { Chip(service: serviceNames[$0.label] ?? $0.label, port: $0.port, up: $0.up) }

        // context: latest non-nil context against the hard cap
        let total = snap.maxInput > 0 ? Double(snap.maxInput) : Double(max(ctxCur, 1))
        var ttft = "–"
        var missed = "–"
        if let m = Self.doneRx.firstMatch(in: snap.main.summary,
                                          range: NSRange(snap.main.summary.startIndex...,
                                                         in: snap.main.summary)) {
            func group(_ i: Int) -> String? {
                guard let r = Range(m.range(at: i), in: snap.main.summary) else { return nil }
                return String(snap.main.summary[r])
            }
            if let ps = group(2), let cs = group(3), let pi = Int(ps), let ci = Int(cs) {
                missed = tokens(pi - ci)
            }
            if let ts = group(4) {
                ttft = String(format: "%.1fs", Double(ts) ?? 0)
            }
        }
        let activeTxt = snap.main.active.map(String.init) ?? "–"
        let ctx = CtxCol(barFrac: min(1, Double(ctxCur) / total),
                         barSuffix: "] \(tokens(ctxCur))/\(tokens(Int(total)))",
                         samples: ringCtx.values,
                         scale: dynamicScale(ringCtx.values, minimum: 4096, step: 2048),
                         capacity: total,
                         current: ctxCur,
                         footer: "TTFT \(ttft) · uncached \(missed) · active \(activeTxt)")
        let smallScale = dynamicScale(ringCtxS.values, minimum: 4096, step: 2048)
        let ctxSmall = CtxCol(barFrac: min(1, Double(ctxSmallCur) / smallScale),
                              barSuffix: "] \(tokens(ctxSmallCur))",
                              samples: ringCtxS.values,
                              scale: smallScale,
                              capacity: total,
                              current: ctxSmallCur,
                              footer: "Small model context")
        let activeSmallRequests = snap.small?.active

        // process table + RAM ledger
        var procRows: [Row] = snap.processes.map { p in
            Row(text: rpad(p.name, 14) + " pid " + rpad(String(p.pid), 7)
                    + String(format: " %5.1f%% cpu", p.cpu)
                    + "  " + lpad(humanBytes(p.rss), 7) + " rss"
                    + String(format: " %4.1f%% ram", p.rssPercent)
                    + "  \(p.threads) thr",
                 color: heat(min(100, p.cpu)))
        }
        if !procRows.isEmpty {
            procRows.append(Row(text: "llm \(humanBytes(snap.processSummary.ours))"
                    + String(format: " · %.0f%% of RAM", snap.processSummary.oursPercent)
                    + " · others \(humanBytes(snap.processSummary.others))"
                    + " · available \(humanBytes(snap.memory.available))",
                 color: heat(snap.memory.percent)))
        } else {
            procRows.append(Row(text: "no model processes found", color: C.red))
        }

        // memory / swap
        let low = snap.memory.available < 10 * 1_073_741_824   // below 10 GB free a Metal eval can OOM
        let memSuffix = low
            ? String(format: "] %3.0f%%  ⚠ OOM risk: %@ free",
                     snap.memory.percent, humanBytes(snap.memory.available))
            : String(format: "] %3.0f%%  %@/%@",
                     snap.memory.percent, humanBytes(snap.memory.used),
                     humanBytes(snap.memory.total))
        var swapFrac: Double?
        var swapSuffix: String?
        if let sw = snap.swap, sw.total > 0 {
            swapFrac = sw.percent / 100
            swapSuffix = String(format: "] %3.0f%%  %@/%@", sw.percent,
                                humanBytes(sw.used), humanBytes(sw.total))
        }

        // heaviest third-party apps — the kill candidates
        let topMem: Row? = snap.topEaters.isEmpty
            ? nil
            : Row(text: " top mem (kill <pid>): "
                    + snap.topEaters.map { "\($0.name) \(humanBytes($0.rss))(\($0.pid))" }
                        .joined(separator: " · "),
                  color: low ? C.red : C.dim)

        // errors red, then the dim tail
        let errorCount = snap.logs.filter(\.error).count
        var logRows = snap.logs.map {
            Row(text: "[\($0.tag)] \($0.text)", color: $0.error ? C.red : C.dim)
        }
        if logRows.isEmpty {
            logRows = [Row(text: "no errors · logs empty", color: C.green)]
        }

        derived = Derived(
            model: snap.model,
            smallModel: snap.smallModel,
            uptime: fmtUptime(snap.uptimeSeconds),
            configHome: snap.configHome ?? (FileManager.default.homeDirectoryForCurrentUser.path + "/.mlx-stack"),
            chips: chips,
            coreCount: snap.cpu.cores.count,
            tokMain: tokCol(tokMain, ringTok),
            tokSmall: snap.ports.small != nil ? tokCol(tokSmall, ringTokS) : nil,
            requestsMain: requestCol(requestMain, ringRequests, uptime: snap.uptimeSeconds),
            requestsSmall: snap.ports.small != nil
                ? requestCol(requestSmall, ringRequestsS, uptime: snap.uptimeSeconds)
                : nil,
            activityMain: activityCol(ringActivity),
            activitySmall: snap.ports.small != nil ? activityCol(ringActivityS) : nil,
            activeRequests: snap.main.active,
            activeRequestsSmall: activeSmallRequests,
            ctx: ctx,
            ctxSmall: ctxSmall,
            cpuTotal: snap.cpu.total,
            cpuSuffix: String(format: "] %5.1f%%", snap.cpu.total),
            cpuHist: ringCpu.values,
            cores: snap.cpu.cores,
            clients: snap.clients,
            procRows: procRows,
            othersBytes: snap.processSummary.others,
            memFrac: snap.memory.percent / 100,
            memSuffix: memSuffix,
            memoryDetail: "Used \(humanBytes(snap.memory.used)) of \(humanBytes(snap.memory.total)) · \(humanBytes(snap.memory.available)) available",
            memLow: low,
            swapFrac: swapFrac,
            swapSuffix: swapSuffix,
            topMem: topMem,
            logRows: logRows,
            errorCount: errorCount)
        if let error = snap.launchError, error.model == loadingModel {
            loadingModel = nil
            loadError = error
        }
        if loadingModel == snap.model, snap.ports.main.up {
            loadingModel = nil
            loadError = nil
        }
    }
}

// MARK: - formatting (same shapes as the backend's ui.py)

enum C {
    static let green = Color(red: 0.30, green: 0.85, blue: 0.45)
    static let yellow = Color(red: 0.95, green: 0.78, blue: 0.30)
    static let red = Color(red: 0.96, green: 0.38, blue: 0.34)
    static let cyan = Color(red: 0.38, green: 0.85, blue: 0.95)
    static let fg = Color(white: 0.86)
    static let dim = Color(white: 0.60)
    static let barBg = Color(white: 0.24)
}

func heat(_ pct: Double) -> Color {   // green -> yellow -> red
    pct < 60 ? C.green : pct < 85 ? C.yellow : C.red
}

func percent(_ fraction: Double) -> String {
    String(format: "%.0f%%", fraction * 100)
}

func copyText(_ text: String) {
    NSPasteboard.general.clearContents()
    NSPasteboard.general.setString(text, forType: .string)
}

func mono(_ size: CGFloat = 12, _ weight: Font.Weight = .regular) -> Font {
    .system(size: size, weight: weight, design: .monospaced)
}

/// ui.human_bytes: B/K/M as integers, G with one decimal.
func humanBytes(_ n: Int64) -> String {
    var v = Double(n)
    for u in ["B", "K", "M"] {
        if v < 1024 {
            return String(format: "%.0f%@", v, u)
        }
        v /= 1024
    }
    return String(format: "%.1fG", v)
}

/// Compact token counts: 999 -> "999", 1500 -> "2K", 15000 -> "15.0K".
func tokens(_ n: Int) -> String {
    if n >= 10000 { return String(format: "%.1fK", Double(n) / 1000) }
    if n >= 1000 { return String(format: "%.0fK", Double(n) / 1000) }
    return String(n)
}

func fmtUptime(_ s: Int) -> String {
    let h = s / 3600, m = (s % 3600) / 60, sec = s % 60
    return h > 0 ? String(format: "%d:%02d:%02d", h, m, sec)
                 : String(format: "%d:%02d", m, sec)
}

func durationText(_ seconds: Double) -> String {
    seconds < 60 ? String(format: "%.1fs", seconds)
                 : String(format: "%.1fm", seconds / 60)
}

func dynamicScale(_ samples: [Double], minimum: Double, step: Double,
                  headroom: Double = 1.15) -> Double {
    let maxValue = samples.max() ?? 0
    if maxValue <= minimum { return minimum }
    let target = max(minimum, maxValue * headroom)
    return max(minimum, (target / step).rounded(.up) * step)
}

func rpad(_ s: String, _ n: Int) -> String {
    s.count >= n ? s : s + String(repeating: " ", count: n - s.count)
}

func lpad(_ s: String, _ n: Int) -> String {
    s.count >= n ? s : String(repeating: " ", count: n - s.count) + s
}

// MARK: - panel view

struct AppRootView: View {
    @ObservedObject var panel: PanelModel
    @ObservedObject var manager: ModelManagerModel
    @ObservedObject var navigation: AppNavigation

    var body: some View {
        Group {
            switch navigation.screen {
            case .models:
                ModelManagerView(model: manager, panel: panel, navigation: navigation)
            case .monitor:
                PanelView(model: panel, navigation: navigation)
            case .tools:
                ToolsView(model: panel, navigation: navigation)
            }
        }
        .onChange(of: panel.derived?.model) { _, name in
            guard let name, name != "no model loaded", !navigation.hasChosenScreen else { return }
            navigation.screen = .monitor
        }
    }
}

struct ScreenPicker: View {
    @ObservedObject var navigation: AppNavigation

    var body: some View {
        Picker("Section", selection: Binding(
            get: { navigation.screen },
            set: { navigation.show($0) }
        )) {
            Text("Models").tag(AppNavigation.Screen.models)
            Text("Monitor").tag(AppNavigation.Screen.monitor)
            Text("Tools").tag(AppNavigation.Screen.tools)
        }
        .pickerStyle(.segmented)
        .labelsHidden()
        .frame(width: 230)
        .accessibilityLabel("Application section")
    }
}

struct PanelView: View {
    @ObservedObject var model: PanelModel
    @ObservedObject var navigation: AppNavigation

    var body: some View {
        ZStack {
            VisualEffectBackground()
            Color.black.opacity(0.55)
            if model.derived != nil, !model.offline {
                VStack(spacing: 0) {
                    MonitorHeader(model: model, navigation: navigation)
                    ScrollViewReader { proxy in
                        ScrollView {
                            PanelContent(model: model, navigation: navigation,
                                         showLogs: {
                                             model.showLogs = true
                                             proxy.scrollTo("monitor-logs", anchor: .bottom)
                                         })
                                .padding(14)
                        }
                    }
                }
            } else {
                OfflineView(model: model, navigation: navigation)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
        .contextMenu {
            Button("Quit MLX Stack") {
                NSApplication.shared.terminate(nil)
            }
            .keyboardShortcut("q", modifiers: .command)
        }
    }
}

struct MonitorHeader: View {
    @ObservedObject var model: PanelModel
    @ObservedObject var navigation: AppNavigation
    private var d: Derived { model.derived! }

    var body: some View {
        VStack(alignment: .leading, spacing: 9) {
            HStack {
                ScreenPicker(navigation: navigation)
                Spacer()
                Text("uptime \(d.uptime)")
                    .font(mono(11))
                    .foregroundStyle(C.dim)
            }
            HStack(spacing: 18) {
                ModelIdentity(role: "Main", name: d.model, active: d.model != "no model loaded",
                              activity: model.selectedModel == .main ? activityState : nil)
                ModelIdentity(role: "Small", name: d.smallModel ?? "not loaded",
                              active: d.smallModel != nil,
                              activity: model.selectedModel == .small ? activityState : nil)
            }
            HStack(spacing: 12) {
                ForEach(Array(d.chips.enumerated()), id: \.offset) { _, chip in
                    ServiceStatusChip(chip: chip)
                }
                Spacer()
                Text("User traffic only")
                    .font(mono(10))
                    .foregroundStyle(C.dim)
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.black.opacity(0.20))
    }
    private var selectedTok: TokCol {
        model.selectedModel == .small ? (d.tokSmall ?? d.tokMain) : d.tokMain
    }

    private var selectedActive: Int {
        model.selectedModel == .small ? (d.activeRequestsSmall ?? 0) : (d.activeRequests ?? 0)
    }

    private var activityState: ActivityState {
        if d.model == "no model loaded" && d.smallModel == nil { return .noModel }
        if selectedActive > 0 { return .active }
        if selectedTok.cur > 0 && selectedTok.txt.contains("prefill") { return .prefilling }
        if selectedTok.cur > 0 { return .generating }
        return .idle
    }
}


private struct ModelIdentity: View {
    let role: String
    let name: String
    let active: Bool
    let activity: ActivityState?

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack(spacing: 6) {
                Text(role).foregroundStyle(C.dim)
                if let activity {
                    ActivityBadge(state: activity)
                }
            }
            Text(name)
                .foregroundStyle(active ? C.fg : C.dim)
                .lineLimit(1)
                .truncationMode(.middle)
                .textSelection(.enabled)
        }
        .font(mono(11))
        .frame(maxWidth: .infinity, alignment: .leading)
        .accessibilityElement(children: .combine)
    }
}

private enum ActivityState {
    case idle, generating, prefilling, active, noModel

    var title: String {
        switch self {
        case .idle: return "Idle"
        case .generating: return "Generating"
        case .prefilling: return "Prefilling"
        case .active: return "Active requests"
        case .noModel: return "No model"
        }
    }

    var icon: String {
        switch self {
        case .idle: return "checkmark.circle"
        case .generating: return "waveform.path.ecg"
        case .prefilling: return "hourglass"
        case .active: return "arrow.triangle.2.circlepath"
        case .noModel: return "circle.dashed"
        }
    }

    var color: Color {
        switch self {
        case .idle: return C.dim
        case .generating: return C.green
        case .prefilling: return C.yellow
        case .active: return C.cyan
        case .noModel: return C.dim
        }
    }
}

private struct ActivityBadge: View {
    let state: ActivityState

    var body: some View {
        Label(state.title, systemImage: state.icon)
            .font(.system(size: 10, weight: .medium))
            .foregroundStyle(state.color)
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(state.color.opacity(0.10), in: RoundedRectangle(cornerRadius: 5))
            .accessibilityLabel("Main model \(state.title.lowercased())")
    }
}

private struct ServiceStatusChip: View {
    let chip: Chip

    var body: some View {
        HStack(spacing: 5) {
            Image(systemName: chip.up ? "checkmark.circle.fill" : "xmark.circle.fill")
                .accessibilityHidden(true)
            // verbatim: a Text literal formats an Int with digit grouping ("19 797")
            Text(verbatim: "\(chip.service) :\(chip.port)")
        }
        .font(.system(size: 11, weight: .medium))
        .foregroundStyle(chip.up ? C.green : C.red)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(Text(verbatim: "\(chip.service) \(chip.up ? "running" : "stopped") on port \(chip.port)"))
    }
}

struct LogRowsView: View {
    let rows: [Row]
    let followsTail: Bool

    var body: some View {
        VStack(spacing: 2) {
            HStack {
                Spacer()
                Button("Copy all logs") {
                    copyText(rows.map(\.text).joined(separator: "\n"))
                }
                .controlSize(.small)
            }
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 3) {
                        ForEach(Array(rows.enumerated()), id: \.offset) { _, row in
                            Text(row.text)
                                .font(mono(11))
                                .foregroundStyle(row.color)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .fixedSize(horizontal: false, vertical: true)
                                .textSelection(.enabled)
                                .contextMenu {
                                    Button("Copy log line") { copyText(row.text) }
                                    Button("Copy all logs") {
                                        copyText(rows.map(\.text).joined(separator: "\n"))
                                    }
                                }
                        }
                        Color.clear.frame(height: 1).id("log-bottom")
                    }
                    .padding(.vertical, 2)
                }
                .onAppear { proxy.scrollTo("log-bottom", anchor: .bottom) }
                .onChange(of: rows.count) {
                    if followsTail { proxy.scrollTo("log-bottom", anchor: .bottom) }
                }
            }
        }
    }
}

struct OfflineView: View {
    @ObservedObject var model: PanelModel
    @ObservedObject var navigation: AppNavigation

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                ScreenPicker(navigation: navigation)
                Spacer()
                Text("Monitor")
                    .font(.title3.weight(.semibold))
                    .foregroundStyle(C.fg)
            }
            Spacer()
            VStack(spacing: 12) {
                Text(model.everConnected ? "Monitor is offline" : "Starting the local backend…")
                    .font(.title3.weight(.semibold))
                    .foregroundStyle(C.fg)
                Text(model.everConnected
                     ? "MLX Stack will reconnect automatically. Your models and settings are unchanged."
                     : "The first launch prepares Python dependencies. This can take a few minutes.")
                    .font(.callout)
                    .foregroundStyle(C.dim)
                    .multilineTextAlignment(.center)
                ProgressView()
                HStack {
                    Button("Open Models") { navigation.show(.models) }
                    Button("Show Log in Finder") {
                        let log = FileManager.default.homeDirectoryForCurrentUser
                            .appendingPathComponent(".mlx-stack/logs/mlx-stack.log")
                        NSWorkspace.shared.activateFileViewerSelecting([log])
                    }
                }
            }
            .padding(24)
            .frame(maxWidth: .infinity)
            .background(Color.white.opacity(0.04), in: RoundedRectangle(cornerRadius: 10))
            Spacer()
            Text("⌘1 Models · ⌘2 Monitor · ⌘3 Tools · ⌘Q Quit")
                .font(mono(11))
                .foregroundStyle(C.dim)
                .lineLimit(1)
        }
        .padding(16)
    }
}

struct PanelContent: View {
    @ObservedObject var model: PanelModel
    @ObservedObject var navigation: AppNavigation
    let showLogs: () -> Void
    private var d: Derived { model.derived! }

    private var hasMain: Bool { d.model != "no model loaded" }
    private var hasSmall: Bool { d.smallModel != nil }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            notices
            if !hasMain && !hasSmall {
                emptyGeneration
            } else {
                performanceSummary
                mainCharts
                systemResources
            }
            MonitorDisclosure(title: "Logs", isExpanded: $model.showLogs) {
                LogRowsView(rows: d.logRows, followsTail: false)
                    .frame(height: 180)
            }
            .id("monitor-logs")
            Text("⌘1 Models · ⌘2 Monitor · ⌘3 Tools · ⌘Q Quit")
                .font(mono(11))
                .foregroundStyle(C.dim)
                .lineLimit(1)
        }
    }

    @ViewBuilder private var notices: some View {
        if let error = model.loadError {
            MonitorNotice(text: "Model launch failed (status \(error.status)). Review the logs before retrying.",
                          actionTitle: "Review logs", action: showLogs)
        }
        if d.errorCount > 0 {
            MonitorNotice(text: "\(d.errorCount) recent backend errors need attention.",
                          actionTitle: "Review logs", action: showLogs)
        }
    }

    private var emptyGeneration: some View {
        Box(" Generation ") {
            VStack(spacing: 8) {
                Text("No model is active")
                    .font(.callout.weight(.semibold))
                    .foregroundStyle(C.fg)
                Text("Load a model to see generation speed, request activity and context.")
                    .font(.callout)
                    .foregroundStyle(C.dim)
                Button("Choose a model") { navigation.show(.models) }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.small)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 18)
        }
    }

    private var showingSmall: Bool { model.selectedModel == .small && hasSmall }
    private var selectedTok: TokCol { showingSmall ? (d.tokSmall ?? d.tokMain) : d.tokMain }
    private var selectedRequests: RequestCol { showingSmall ? (d.requestsSmall ?? d.requestsMain) : d.requestsMain }
    private var selectedCtx: CtxCol { showingSmall ? (d.ctxSmall ?? d.ctx) : d.ctx }
    private var selectedActive: Int { showingSmall ? (d.activeRequestsSmall ?? 0) : (d.activeRequests ?? 0) }
    private var selectedActivity: ActivityCol { showingSmall ? (d.activitySmall ?? d.activityMain) : d.activityMain }

    private var performanceSummary: some View {
        let contextValue = selectedCtx.barSuffix.replacingOccurrences(of: "] ", with: "")
        let contextDetail = selectedCtx.footer
        return HStack(alignment: .top, spacing: 0) {
            PerformanceSummaryItem(
                title: showingSmall ? "Small model" : "Real output",
                value: selectedTok.cur > 0 ? String(format: "%.1f", selectedTok.cur) : "idle",
                detail: selectedTok.cur > 0
                    ? selectedTok.txt.trimmingCharacters(in: .whitespaces)
                    : "waiting for a request",
                footer: selectedTok.avg.map {
                    String(format: "avg %.0f · peak %.0f tok/s while generating", $0, selectedTok.peak)
                } ?? "no generation yet",
                valueColor: selectedTok.cur > 0 ? C.fg : C.dim)
            Rectangle().fill(C.dim.opacity(0.25)).frame(width: 1, height: 64)
            PerformanceSummaryItem(
                title: "Requests",
                value: "\(selectedRequests.total)",
                detail: selectedRequests.total > 0
                    ? String(format: "%.1f per minute · %d active", selectedRequests.perMinute, selectedActive)
                    : "none since load · \(selectedActive) active",
                footer: requestActivity(selectedRequests))
            Rectangle().fill(C.dim.opacity(0.25)).frame(width: 1, height: 64)
            PerformanceSummaryItem(
                title: "Context",
                value: contextValue,
                detail: contextDetail,
                footer: showingSmall ? "Small model context" : "Context window and request latency")
            Rectangle().fill(C.dim.opacity(0.25)).frame(width: 1, height: 64)
            PerformanceSummaryItem(
                title: "Memory",
                value: String(format: "%.0f%%", d.memFrac * 100),
                detail: d.memoryDetail,
                footer: d.swapSuffix.map { "Swap \($0.replacingOccurrences(of: "] ", with: ""))" }
                    ?? "Swap not reported",
                valueColor: d.memLow ? C.red : C.fg)
        }
        .padding(10)
        .background(Color.white.opacity(0.04), in: RoundedRectangle(cornerRadius: 8))
    }

    private func requestActivity(_ requests: RequestCol) -> String {
        let last = requests.lastAgo.map { String(format: "Last %.0fs ago", $0) }
            ?? "No real requests yet"
        let average = requests.averageInterval.map { "avg every \(durationText($0))" }
            ?? "avg interval pending"
        return last + " · " + average
    }

    private var contextHistorySummary: String {
        let values = selectedCtx.samples.filter { $0 >= 0 }
        guard !values.isEmpty else { return "No context history yet" }
        let low = values.min() ?? 0
        let high = values.max() ?? 0
        if high - low <= max(1, selectedCtx.scale * 0.01) {
            return "Stable at \(tokens(selectedCtx.current)) · auto 0–\(tokens(Int(selectedCtx.scale))) · cap \(tokens(Int(selectedCtx.capacity)))"
        }
        return "Range \(tokens(Int(low)))–\(tokens(Int(high))) · auto 0–\(tokens(Int(selectedCtx.scale))) · cap \(tokens(Int(selectedCtx.capacity)))"
    }

    private var modelSelector: some View {
        HStack {
            Text("Performance")
                .font(.callout.weight(.medium))
                .foregroundStyle(C.fg)
            Spacer()
            Picker("Model", selection: $model.selectedModel) {
                if hasMain {
                    Text("Main").tag(MonitorModel.main)
                }
                if hasSmall {
                    Text("Small").tag(MonitorModel.small)
                }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .frame(width: 140)
        }
        .onAppear {
            if !hasMain && hasSmall { model.selectedModel = .small }
        }
    }

    private var mainCharts: some View {
        VStack(alignment: .leading, spacing: 8) {
            modelSelector
            HStack(alignment: .top, spacing: 12) {
                Box(" Generation ") {
                    VStack(alignment: .leading, spacing: 6) {
                        RateChart(samples: selectedTok.samples, scale: selectedTok.scale,
                                  colored: { _ in C.green },
                                  summary: selectedTok.txt)
                            .frame(maxHeight: .infinity)
                        ActivityBand(activity: selectedActivity.samples,
                                     requests: selectedRequests.samples,
                                     summary: activityCaption)
                        TimeAxis()
                        Text(activityCaption)
                            .font(mono(10))
                            .foregroundStyle(C.dim)
                            .lineLimit(1)
                        Text(selectedTok.footer)
                            .font(mono(10))
                            .foregroundStyle(C.dim)
                            .lineLimit(2)
                    }
                }
                .frame(maxHeight: .infinity, alignment: .top)
                Box(" Context usage ") {
                    VStack(alignment: .leading, spacing: 6) {
                        TermBar(label: " Used ", frac: selectedCtx.barFrac, suffix: selectedCtx.barSuffix)
                        RateChart(samples: selectedCtx.samples, scale: selectedCtx.scale,
                                  colored: { heat(100 * $0 / max(selectedCtx.capacity, 1)) },
                                  tickLabel: { tokens(Int($0)) },
                                  summary: selectedCtx.barSuffix.replacingOccurrences(of: "] ", with: ""))
                            .frame(maxHeight: .infinity)
                        TimeAxis()
                        Text(contextHistorySummary)
                            .font(mono(10))
                            .foregroundStyle(C.dim)
                            .lineLimit(1)
                        Text(selectedCtx.footer)
                            .font(mono(10))
                            .foregroundStyle(C.dim)
                            .lineLimit(2)
                    }
                }
                .frame(maxHeight: .infinity, alignment: .top)
            }
            .frame(height: 240)
            .fixedSize(horizontal: false, vertical: true)
        }
    }

    /// The chart's main message in words (HIG): where the model's time went.
    private var activityCaption: String {
        let activity = selectedActivity
        guard !activity.samples.isEmpty else { return "No history yet" }
        let seconds = activity.samples.count
        let window = seconds >= 120 ? String(format: "%.0f min", Double(seconds) / 60)
            : seconds >= 60 ? "1 min" : "\(seconds) s"
        let count = Int(selectedRequests.samples.reduce(0, +))
        return "generating \(percent(activity.generating)) · reading prompt \(percent(activity.reading))"
            + " · waiting for client \(percent(activity.waiting))"
            + " · \(count) request\(count == 1 ? "" : "s") in \(window)"
    }

    private var systemResources: some View {
        Box(" System resources ") {
            HStack(alignment: .top, spacing: 16) {
                VStack(alignment: .leading, spacing: 7) {
                    TermBar(label: " CPU ", frac: d.cpuTotal / 100, suffix: d.cpuSuffix)
                    RateChart(samples: d.cpuHist, scale: 100, colored: { heat($0) },
                              tickLabel: { $0 > 0 ? "\(Int($0))%" : "0" },
                              summary: "CPU use over the last four minutes")
                        .frame(height: 48)
                    Text("Last 4 minutes · \(d.coreCount) cores")
                        .font(mono(10))
                        .foregroundStyle(C.dim)
                    MonitorDisclosure(title: "CPU cores", isExpanded: $model.showCPUCores) {
                        coresGrid
                    }
                }
                Rectangle().fill(C.dim.opacity(0.25)).frame(width: 1)
                VStack(alignment: .leading, spacing: 7) {
                    TermBar(label: " RAM ", frac: d.memFrac, suffix: d.memSuffix,
                            fill: d.memLow ? C.red : nil,
                            suffixColor: d.memLow ? C.red : C.fg)
                    if let swSuffix = d.swapSuffix {
                        TermBar(label: " SWP ", frac: d.swapFrac ?? 0, suffix: swSuffix)
                    }
                    if let tm = d.topMem {
                        Text(tm.text)
                            .font(mono(10))
                            .foregroundStyle(tm.color)
                            .lineLimit(2)
                    }
                    MonitorDisclosure(title: "Processes", isExpanded: $model.showProcesses) {
                        VStack(alignment: .leading, spacing: 3) {
                            ForEach(Array(d.procRows.enumerated()), id: \.offset) { _, row in
                                Text(row.text)
                                    .font(mono(10))
                                    .foregroundStyle(row.color)
                                    .lineLimit(1)
                            }
                        }
                    }
                }
            }
        }
    }

    private var coresGrid: some View {
        LazyVGrid(columns: [GridItem(.adaptive(minimum: 105, maximum: 140), spacing: 10)],
                  alignment: .leading, spacing: 7) {
            ForEach(Array(d.cores.enumerated()), id: \.offset) { i, value in
                HStack(spacing: 5) {
                    Text(String(format: "C%02d", i))
                        .foregroundStyle(heat(value))
                        .frame(width: 24, alignment: .leading)
                    MiniBar(frac: value / 100)
                        .frame(width: 28, height: 6)
                    Text(String(format: "%3.0f%%", value))
                        .foregroundStyle(heat(value))
                        .frame(width: 32, alignment: .trailing)
                }
                .font(mono(10))
                .frame(maxWidth: .infinity, alignment: .leading)
                .accessibilityElement(children: .ignore)
                .accessibilityLabel("CPU core \(i)")
                .accessibilityValue(String(format: "%.0f percent", value))
            }
        }
}
}

private struct MonitorNotice: View {
    let text: String
    let actionTitle: String
    let action: () -> Void

    var body: some View {
        HStack(spacing: 9) {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(C.red)
                .accessibilityHidden(true)
            Text(text)
                .font(.callout)
                .foregroundStyle(C.fg)
                .frame(maxWidth: .infinity, alignment: .leading)
            Button(actionTitle, action: action)
        }
        .padding(10)
        .background(C.red.opacity(0.10), in: RoundedRectangle(cornerRadius: 8))
    }
}


private struct PerformanceSummaryItem: View {
    let title: String
    let value: String
    let detail: String
    let footer: String
    var valueColor: Color = C.fg

    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(title)
                .font(.callout)
                .foregroundStyle(C.dim)
            Text(value)
                .font(mono(20, .semibold))
                .foregroundStyle(valueColor)
                .lineLimit(1)
                .minimumScaleFactor(0.7)
            Text(detail)
                .font(mono(10))
                .foregroundStyle(C.fg)
                .lineLimit(2)
            Text(footer)
                .font(mono(9))
                .foregroundStyle(C.dim)
                .lineLimit(2)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, 12)
        .accessibilityElement(children: .combine)
    }
}


private struct MonitorDisclosure<Content: View>: View {
    let title: String
    @Binding var isExpanded: Bool
    let content: Content

    init(title: String, isExpanded: Binding<Bool>, @ViewBuilder content: () -> Content) {
        self.title = title
        self._isExpanded = isExpanded
        self.content = content()
    }

    var body: some View {
        DisclosureGroup(title, isExpanded: $isExpanded) {
            content
                .padding(.top, 7)
        }
        .font(.callout)
    }
}

// MARK: - terminal primitives

/// label[■■■■■□□□□] suffix.
struct TermBar: View {
    let label: String
    let frac: Double
    let suffix: String
    var fill: Color? = nil       // nil = heat(frac * 100)
    var suffixColor: Color = C.fg

    var body: some View {
        HStack(spacing: 0) {
            Text(label)
                .font(mono(12, .semibold))
                .foregroundStyle(C.fg)
            GeometryReader { g in
                ZStack(alignment: .leading) {
                    Rectangle().fill(C.barBg)
                    Rectangle()
                        .fill(fill ?? heat(frac * 100))
                        .frame(width: max(0, min(1, frac)) * g.size.width)
                }
            }
            .frame(height: 12)
            Text(suffix)
                .font(mono(12, .semibold))
                .foregroundStyle(suffixColor)
                .lineLimit(1)
        }
        .fixedSize(horizontal: false, vertical: true)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(label.replacingOccurrences(of: "[", with: "")
            .trimmingCharacters(in: .whitespaces))
        .accessibilityValue(suffix.replacingOccurrences(of: "] ", with: ""))
    }
}

struct MiniBar: View {
    let frac: Double

    var body: some View {
        GeometryReader { g in
            ZStack(alignment: .leading) {
                Rectangle().fill(C.barBg)
                Rectangle()
                    .fill(heat(frac * 100))
                    .frame(width: max(0, min(1, frac)) * g.size.width)
            }
        }
        .accessibilityHidden(true)
    }
}

// MARK: - history charts

// Every history is a ring of `historySeconds` one-second samples, and every
// chart maps time the same way: one slot per second with "now" at the right
// edge, so a short history grows in from the right instead of sliding along.

enum ChartAxis {
    /// Width of the value-axis gutter on the leading edge of every chart.
    static let gutter: CGFloat = 40
}

/// A time series as a line with a soft fill and a three-tick value axis.
struct RateChart: View {
    let samples: [Double]
    let scale: Double
    let colored: (Double) -> Color
    var tickLabel: (Double) -> String = { String(format: "%.0f", $0) }
    var summary: String = "Rate over the last four minutes"

    var body: some View {
        Canvas { ctx, size in
            let plot = CGRect(x: ChartAxis.gutter, y: 7,
                              width: size.width - ChartAxis.gutter, height: size.height - 10)
            guard plot.width > 0, plot.height > 0 else { return }
            for fraction in [0.0, 0.5, 1.0] {
                let y = plot.maxY - plot.height * CGFloat(fraction)
                var grid = Path()
                grid.move(to: CGPoint(x: plot.minX, y: y))
                grid.addLine(to: CGPoint(x: plot.maxX, y: y))
                ctx.stroke(grid, with: .color(C.dim.opacity(fraction == 0 ? 0.3 : 0.12)),
                           lineWidth: 1)
                ctx.draw(Text(tickLabel(scale * fraction)).font(mono(9)).foregroundColor(C.dim),
                         at: CGPoint(x: plot.minX - 7, y: y), anchor: .trailing)
            }
            let n = samples.count
            guard n > 0, scale > 0 else { return }
            let step = plot.width / CGFloat(historySeconds)
            func point(_ i: Int) -> CGPoint {
                CGPoint(x: plot.maxX - CGFloat(n - 1 - i) * step - step / 2,
                        y: plot.maxY - CGFloat(max(0, min(1, samples[i] / scale))) * plot.height)
            }
            var previous = point(0)
            if n == 1 {
                ctx.fill(Path(ellipseIn: CGRect(x: previous.x - 2, y: previous.y - 2,
                                                width: 4, height: 4)),
                         with: .color(colored(samples[0])))
            }
            for i in 1..<n {
                let current = point(i)
                let color = colored(samples[i])
                var area = Path()
                area.move(to: previous)
                area.addLine(to: current)
                area.addLine(to: CGPoint(x: current.x, y: plot.maxY))
                area.addLine(to: CGPoint(x: previous.x, y: plot.maxY))
                area.closeSubpath()
                ctx.fill(area, with: .color(color.opacity(0.12)))
                // a flat run of zeros is an idle gap: leave it to the axis
                if samples[i] > 0 || samples[i - 1] > 0 {
                    var line = Path()
                    line.move(to: previous)
                    line.addLine(to: current)
                    ctx.stroke(line, with: .color(color),
                               style: StrokeStyle(lineWidth: 1.5, lineCap: .round, lineJoin: .round))
                }
                previous = current
            }
            if samples[n - 1] > 0 {
                let dot = CGRect(x: previous.x - 2.5, y: previous.y - 2.5, width: 5, height: 5)
                ctx.fill(Path(ellipseIn: dot), with: .color(colored(samples[n - 1])))
            }
        }
        .accessibilityLabel("History")
        .accessibilityValue(summary)
    }
}

/// Time labels under a history chart: four minutes, "now" at the right edge.
struct TimeAxis: View {
    var body: some View {
        HStack(spacing: 0) {
            Color.clear.frame(width: ChartAxis.gutter, height: 1)
            HStack {
                Text("4 min ago")
                Spacer()
                Text("2 min ago")
                Spacer()
                Text("now")
            }
        }
        .font(mono(9))
        .foregroundStyle(C.dim)
        .frame(height: 11)
        .accessibilityHidden(true)
    }
}

/// What the model did each second: generating (full height), reading a
/// prompt (a lower band) or waiting for the client (the empty track), with
/// a tick for every second in which a request arrived.
struct ActivityBand: View {
    let activity: [Double]
    let requests: [Double]
    var summary: String = "Model activity over the last four minutes"

    var body: some View {
        Canvas { ctx, size in
            let n = activity.count
            guard n > 0 else { return }
            let step = (size.width - ChartAxis.gutter) / CGFloat(historySeconds)
            let origin = size.width - CGFloat(n) * step
            ctx.fill(Path(CGRect(x: ChartAxis.gutter, y: 0, width: size.width - ChartAxis.gutter,
                                 height: size.height)), with: .color(C.barBg.opacity(0.6)))
            for (i, state) in activity.enumerated() where state != Activity.idle {
                let height = state == Activity.decode ? size.height : size.height * 0.55
                let rect = CGRect(x: origin + CGFloat(i) * step, y: size.height - height,
                                  width: step + 0.5, height: height)
                ctx.fill(Path(rect), with: .color((state == Activity.decode ? C.green : C.yellow).opacity(0.9)))
            }
            for (i, count) in requests.enumerated() where count > 0 && i < n {
                ctx.fill(Path(CGRect(x: origin + CGFloat(i) * step + step / 2 - 1, y: 0,
                                     width: 2, height: 4)), with: .color(C.fg))
            }
        }
        .frame(height: 14)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Model activity")
        .accessibilityValue(summary)
    }
}

/// Bordered box with a title chip.
struct Box<Content: View>: View {
    private let title: String
    private let makeContent: () -> Content

    init(_ title: String, @ViewBuilder content: @escaping () -> Content) {
        self.title = title
        self.makeContent = content
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(mono(11))
                .foregroundStyle(C.dim)
                .padding(.horizontal, 6)
                .padding(.top, 5)
            makeContent()
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.bottom, 7)
        .padding(.horizontal, 4)
        .overlay(
            RoundedRectangle(cornerRadius: 7, style: .continuous)
                .strokeBorder(C.dim.opacity(0.5), lineWidth: 1))
        .clipShape(RoundedRectangle(cornerRadius: 7, style: .continuous))
    }
}
