import AppKit
import SwiftUI

struct ToolsView: View {
    @ObservedObject var model: PanelModel
    @ObservedObject var navigation: AppNavigation
    private var d: Derived? { model.derived }

    var body: some View {
        ZStack {
            VisualEffectBackground()
            Color.black.opacity(0.62)
            VStack(spacing: 0) {
                HStack {
                    ScreenPicker(navigation: navigation)
                    Spacer()
                    if let home = d?.configHome {
                        Button("Open .mlx-stack") { reveal(home) }
                    }
                }
                .padding(18)
                ScrollView {
                    VStack(alignment: .leading, spacing: 14) {
                        Text("Tools")
                            .font(.title2.weight(.semibold))
                            .foregroundStyle(C.fg)
                        Text("Connection settings for local harnesses. Canonical copies live in ~/.mlx-stack/configs; the paths expected by each tool remain symlinks.")
                            .font(.callout)
                            .foregroundStyle(C.dim)
                        if let d {
                            if let backup = model.latestConfigBackup {
                                configBackup(backup)
                            }
                            if let error = model.actionError {
                                Text(error)
                                    .font(.callout)
                                    .foregroundStyle(C.red)
                                    .textSelection(.enabled)
                            }
                            ForEach(Array(d.clients.enumerated()), id: \.offset) { _, client in
                                HarnessConfigRow(client: client, model: model)
                            }
                            stackFiles(home: d.configHome)
                        } else {
                            ProgressView("Loading configuration…")
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }
                    }
                    .padding(18)
                }
            }
        }
        .alert("Restore the latest agent config backup?",
               isPresented: $model.confirmConfigRollback) {
            Button("Cancel", role: .cancel) { }
            Button("Restore", role: .destructive) { model.rollbackLatestConfig() }
        } message: {
            if let backup = model.latestConfigBackup {
                Text("Restore \(backup.model) settings from \(backup.createdAt). Current agent configs will be replaced.")
            }
        }
    }

    private func configBackup(_ backup: Snapshot.ConfigBackup) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Agent configuration backup")
                .font(.headline)
                .foregroundStyle(C.fg)
            Text("\(backup.createdAt) · \(backup.model)")
                .font(.callout)
                .foregroundStyle(C.dim)
            Text(backup.path)
                .font(mono(10))
                .foregroundStyle(C.dim)
                .textSelection(.enabled)
            HStack {
                Button("Show in Finder") { reveal(backup.path) }
                Button("Restore latest backup") { model.confirmConfigRollback = true }
                    .disabled(model.modelActionInProgress)
            }
        }
        .padding(12)
        .background(Color.white.opacity(0.035), in: RoundedRectangle(cornerRadius: 8))
    }

    private func stackFiles(home: String) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Stack configuration")
                .font(.headline)
                .foregroundStyle(C.fg)
            ConfigFileRow(name: "Model defaults", path: home + "/model-settings.json",
                          description: "Per-model context, cache and sampling settings", model: model)
            ConfigFileRow(name: "Global context", path: home + "/context-settings.json",
                          description: "Stack-wide context and prompt-cache limits", model: model)
            ConfigFileRow(name: "omp roles", path: home + "/configs/omp-config.yml",
                          description: "Small-model role used for background calls", model: model)
            ConfigFileRow(name: "Hugging Face token", path: home + "/hf-token",
                          description: "Optional gated-model and download credentials", model: model,
                          secret: true)
            ConfigFileRow(name: "Logs", path: home + "/logs",
                          description: "Backend, proxy and model diagnostics", model: model)
        }
        .padding(.top, 8)
    }

    private func reveal(_ path: String) {
        NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: path)])
    }
}

private struct HarnessConfigRow: View {
    let client: Snapshot.Client
    @ObservedObject var model: PanelModel

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .firstTextBaseline) {
                Text(client.name)
                    .font(.callout.weight(.semibold))
                    .foregroundStyle(C.fg)
                Text(client.role ?? "Local harness")
                    .font(.callout)
                    .foregroundStyle(C.dim)
                Spacer()
                Button {
                    model.copyClient(client.command)
                } label: {
                    Label(model.copiedCommand == client.command ? "Copied" : "Copy command",
                          systemImage: model.copiedCommand == client.command ? "checkmark" : "doc.on.doc")
                }
                .accessibilityLabel("Copy command for \(client.name)")
            }
            Text(client.command)
                .font(mono(11))
                .foregroundStyle(C.fg)
                .textSelection(.enabled)
                .lineLimit(2)
            HStack(spacing: 18) {
                LabeledValue(name: "Model", value: client.model ?? "—")
                LabeledValue(name: "Endpoint", value: client.apiBase ?? "—")
            }
            HStack(spacing: 8) {
                Text(client.configPath ?? "Config path not reported")
                    .font(mono(10))
                    .foregroundStyle(C.dim)
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .textSelection(.enabled)
                Spacer()
                if let path = client.configPath {
                    Button("Show in Finder") { reveal(path) }
                    Button("Copy path") { model.copyClient(path) }
                        .accessibilityLabel("Copy config path for \(client.name)")
                }
            }
        }
        .padding(12)
        .background(Color.white.opacity(0.035), in: RoundedRectangle(cornerRadius: 8))
    }

    private func reveal(_ path: String) {
        NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: path)])
    }
}

private struct ConfigFileRow: View {
    let name: String
    let path: String
    let description: String
    @ObservedObject var model: PanelModel
    var secret = false

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 10) {
            VStack(alignment: .leading, spacing: 3) {
                Text(name)
                    .font(.callout.weight(.medium))
                    .foregroundStyle(C.fg)
                Text(description)
                    .font(.callout)
                    .foregroundStyle(C.dim)
                Text(path)
                    .font(mono(10))
                    .foregroundStyle(C.dim)
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .textSelection(.enabled)
                if secret {
                    Text(FileManager.default.fileExists(atPath: path) ? "Configured" : "Not configured")
                        .font(.callout)
                        .foregroundStyle(C.cyan)
                }
            }
            Spacer()
            Button("Show in Finder") { reveal(path) }
            Button("Copy path") { model.copyClient(path) }
                .accessibilityLabel("Copy path for \(name)")
        }
        .padding(.vertical, 5)
    }

    private func reveal(_ path: String) {
        NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: path)])
    }
}

private struct LabeledValue: View {
    let name: String
    let value: String

    var body: some View {
        HStack(spacing: 5) {
            Text(name).foregroundStyle(C.dim)
            Text(value)
                .foregroundStyle(C.fg)
                .textSelection(.enabled)
                .lineLimit(1)
        }
        .font(.callout)
    }
}
