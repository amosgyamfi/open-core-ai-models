// Community port of kyutai/pocket-tts to Apple Core AI — NOT an Apple model.
// ⚠️ DRAFT — compile on macOS 27. One `.aimodel` graph (`main`) + its descriptor. Loads via
// `PreparedModel.prepare(at:)` — the path that probes structure and gives single-`main`
// dynamic-shape graphs GPU specialization (raw `AIModel(contentsOf:)` defaults to ANE and
// crashes on these graphs; see knowledge/swift-runtime.md).
import CoreAI
import CoreAIShared
import Foundation

final class Bundle: @unchecked Sendable {
    let fn: InferenceFunction
    let descriptor: InferenceFunctionDescriptor

    private init(fn: InferenceFunction, descriptor: InferenceFunctionDescriptor) {
        self.fn = fn
        self.descriptor = descriptor
    }

    static func load(at url: URL, function: String = "main") async throws -> Bundle {
        let prepared = try await PreparedModel.prepare(at: url)
        let model = prepared.model
        guard let desc = model.functionDescriptor(for: function) else {
            throw PocketError.engine("function '\(function)' missing in \(url.lastPathComponent)")
        }
        guard let fn = try model.loadFunction(named: function) else {
            throw PocketError.engine("cannot load '\(function)' in \(url.lastPathComponent)")
        }
        return Bundle(fn: fn, descriptor: desc)
    }

    /// Allocate an input NDArray, resolving any dynamic dims to the requested concrete `shape`.
    func makeInput(_ name: String, shape: [Int]) -> NDArray {
        guard case .ndArray(let d) = descriptor.inputDescriptor(of: name) else {
            preconditionFailure("input \(name) is not an NDArray")
        }
        return NDArray(descriptor: d.resolvingDynamicDimensions(shape))
    }

    func makeOutput(_ name: String, shape: [Int]) -> NDArray {
        guard case .ndArray(let d) = descriptor.outputDescriptor(of: name) else {
            preconditionFailure("output \(name) is not an NDArray")
        }
        return NDArray(descriptor: d.resolvingDynamicDimensions(shape))
    }

    /// Allocate a zero-initialized fp16 state NDArray at full capacity.
    func makeState(_ name: String, fullShape: [Int]) -> NDArray {
        guard case .ndArray(let d) = descriptor.stateDescriptor(of: name) else {
            preconditionFailure("state \(name) is not an NDArray")
        }
        var arr = NDArray(descriptor: d.resolvingDynamicDimensions(fullShape))
        let count = fullShape.reduce(1, *)
        fill16(&arr, count: count) { _ in 0 }
        return arr
    }
}
