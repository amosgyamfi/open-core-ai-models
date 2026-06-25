// Community port of kyutai/pocket-tts to Apple Core AI — NOT an Apple model.
// ⚠️ DRAFT — compile on macOS 27 against the CoreAI framework. Self-contained NDArray
// fill/read helpers (the equivalents are module-internal in Apple's CoreAILanguageModels),
// built against NDArray's public `mutableView(as:)` / `view(as:)` API as used by
// Apple's `CoreAISequentialEngine`.
import CoreAI

/// Fill an `NDArray` of fp16 from a flat `[Float]` (row-major, count == product of shape).
@inlinable
func fill16(_ array: inout NDArray, _ values: [Float]) {
    var view = array.mutableView(as: Float16.self)
    view.withUnsafeMutablePointer { ptr, _, _ in
        for i in 0..<values.count { ptr[i] = Float16(values[i]) }
    }
}

/// Fill `count` fp16 elements from a closure of the flat index.
@inlinable
func fill16(_ array: inout NDArray, count: Int, _ make: (Int) -> Float) {
    var view = array.mutableView(as: Float16.self)
    view.withUnsafeMutablePointer { ptr, _, _ in
        for i in 0..<count { ptr[i] = Float16(make(i)) }
    }
}

/// Fill `count` Int32 elements from a closure of the flat index.
@inlinable
func fill32(_ array: inout NDArray, count: Int, _ make: (Int) -> Int32) {
    var view = array.mutableView(as: Int32.self)
    view.withUnsafeMutablePointer { ptr, _, _ in
        for i in 0..<count { ptr[i] = make(i) }
    }
}

/// Read `count` fp16 elements out of an `NDArray` into `[Float]`.
@inlinable
func read16(_ array: NDArray, count: Int) -> [Float] {
    var out = [Float](); out.reserveCapacity(count)
    array.view(as: Float16.self).withUnsafePointer { ptr, _, _ in
        for i in 0..<count { out.append(Float(ptr[i])) }
    }
    return out
}
