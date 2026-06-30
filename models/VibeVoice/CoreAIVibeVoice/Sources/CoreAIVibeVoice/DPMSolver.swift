// DPMSolverMultistepScheduler — faithful port of the configuration VibeVoice uses:
//   beta_schedule = "cosine" (squaredcos_cap_v2), prediction_type = "v_prediction",
//   algorithm_type = "dpmsolver++", solver_order = 2, solver_type = "midpoint",
//   final_sigmas_type = "zero", timestep_spacing = "linspace", lower_order_final.
//
// Operates on a single latent vector (length = acousticVAEDim). Classifier-free
// guidance is applied by the caller before `step`, so this matches the Python
// `sample_speech_tokens` math exactly (both CFG halves are identical there).

import Foundation

public final class DPMSolverMultistep {
    private let numTrainTimesteps: Int
    private let solverOrder = 2

    // Precomputed full-resolution sigmas over the training schedule.
    private let sigmasFull: [Double]

    // Per-inference-run state.
    private(set) var timesteps: [Int] = []
    private var sigmas: [Double] = []           // length numInferenceSteps + 1
    private var modelOutputs: [[Double]] = []    // ring of last `solverOrder` x0 predictions
    private var lowerOrderNums = 0
    private var stepIndex = 0

    public init(numTrainTimesteps: Int = 1000) {
        self.numTrainTimesteps = numTrainTimesteps

        // Cosine beta schedule (Glide), betas_for_alpha_bar.
        func alphaBar(_ t: Double) -> Double {
            let x = (t + 0.008) / 1.008 * Double.pi / 2.0
            return cos(x) * cos(x)
        }
        var betas = [Double](repeating: 0, count: numTrainTimesteps)
        for i in 0..<numTrainTimesteps {
            let t1 = Double(i) / Double(numTrainTimesteps)
            let t2 = Double(i + 1) / Double(numTrainTimesteps)
            betas[i] = min(1.0 - alphaBar(t2) / alphaBar(t1), 0.999)
        }
        var acp = [Double](repeating: 0, count: numTrainTimesteps)
        var running = 1.0
        for i in 0..<numTrainTimesteps {
            running *= (1.0 - betas[i])
            acp[i] = running
        }
        self.sigmasFull = acp.map { sqrt((1.0 - $0) / $0) }
    }

    /// Configure the discrete timesteps for `n` inference steps (linspace spacing).
    public func setTimesteps(_ n: Int) {
        let lastTimestep = Double(numTrainTimesteps)  // lambda_min_clipped = -inf → clipped_idx 0
        // linspace(0, lastTimestep-1, n+1).round()[::-1][:-1]
        var ts: [Int] = []
        for i in 0...n {
            let v = (lastTimestep - 1.0) * Double(i) / Double(n)
            ts.append(Int(v.rounded()))
        }
        ts.reverse()
        ts.removeLast()
        self.timesteps = ts

        // Interpolate sigmas at the chosen integer timesteps, then append 0.
        var sig = ts.map { t -> Double in interpSigma(Double(t)) }
        sig.append(0.0)
        self.sigmas = sig

        self.modelOutputs = []
        self.lowerOrderNums = 0
        self.stepIndex = 0
    }

    /// Number of configured inference steps.
    public var count: Int { timesteps.count }

    /// The initial standard deviation for the latent noise (init_noise_sigma = 1.0).
    public var initNoiseSigma: Double { 1.0 }

    private func interpSigma(_ t: Double) -> Double {
        // np.interp over arange(numTrainTimesteps) → sigmasFull, at point t.
        if t <= 0 { return sigmasFull[0] }
        if t >= Double(numTrainTimesteps - 1) { return sigmasFull[numTrainTimesteps - 1] }
        let lo = Int(floor(t))
        let hi = lo + 1
        let frac = t - Double(lo)
        return sigmasFull[lo] * (1 - frac) + sigmasFull[hi] * frac
    }

    private func alphaSigma(_ sigma: Double) -> (Double, Double) {
        let alpha = 1.0 / sqrt(sigma * sigma + 1.0)
        return (alpha, sigma * alpha)
    }

    /// v_prediction → x0 prediction (dpmsolver++ data-prediction form).
    private func convertModelOutput(_ modelOutput: [Double], sample: [Double]) -> [Double] {
        let (alphaT, sigmaT) = alphaSigma(sigmas[stepIndex])
        var x0 = [Double](repeating: 0, count: modelOutput.count)
        for i in 0..<x0.count { x0[i] = alphaT * sample[i] - sigmaT * modelOutput[i] }
        return x0
    }

    private func firstOrder(_ m0: [Double], sample: [Double]) -> [Double] {
        let (alphaT, sigmaT) = alphaSigma(sigmas[stepIndex + 1])
        let (_, sigmaS) = alphaSigma(sigmas[stepIndex])
        let lambdaT = log(alphaT) - log(sigmaT)
        let lambdaS = log(1.0 / sqrt(sigmas[stepIndex] * sigmas[stepIndex] + 1.0)) - log(sigmaS)
        let h = lambdaT - lambdaS
        let c1 = sigmaT / sigmaS
        let c2 = alphaT * (exp(-h) - 1.0)
        var out = [Double](repeating: 0, count: m0.count)
        for i in 0..<out.count { out[i] = c1 * sample[i] - c2 * m0[i] }
        return out
    }

    private func secondOrder(_ outputs: [[Double]], sample: [Double]) -> [Double] {
        let m0 = outputs[outputs.count - 1]
        let m1 = outputs[outputs.count - 2]
        let (alphaT, sigmaT) = alphaSigma(sigmas[stepIndex + 1])
        let (alphaS0, sigmaS0) = alphaSigma(sigmas[stepIndex])
        let (alphaS1, sigmaS1) = alphaSigma(sigmas[stepIndex - 1])
        let lambdaT = log(alphaT) - log(sigmaT)
        let lambdaS0 = log(alphaS0) - log(sigmaS0)
        let lambdaS1 = log(alphaS1) - log(sigmaS1)
        let h = lambdaT - lambdaS0
        let h0 = lambdaS0 - lambdaS1
        let r0 = h0 / h
        let cSample = sigmaT / sigmaS0
        let cD0 = alphaT * (exp(-h) - 1.0)
        var out = [Double](repeating: 0, count: m0.count)
        for i in 0..<out.count {
            let d0 = m0[i]
            let d1 = (1.0 / r0) * (m0[i] - m1[i])
            out[i] = cSample * sample[i] - cD0 * d0 - 0.5 * cD0 * d1
        }
        return out
    }

    /// One scheduler step. `modelOutput` is the (CFG-guided) network output.
    public func step(modelOutput: [Float], sample: [Float]) -> [Float] {
        let mo = modelOutput.map { Double($0) }
        let s = sample.map { Double($0) }

        let isFinal = stepIndex == timesteps.count - 1   // final_sigmas_type == "zero"

        let x0 = convertModelOutput(mo, sample: s)
        if modelOutputs.count >= solverOrder { modelOutputs.removeFirst() }
        modelOutputs.append(x0)

        let prev: [Double]
        if lowerOrderNums < 1 || isFinal {
            prev = firstOrder(x0, sample: s)
        } else {
            prev = secondOrder(modelOutputs, sample: s)
        }
        if lowerOrderNums < solverOrder { lowerOrderNums += 1 }
        stepIndex += 1
        return prev.map { Float($0) }
    }
}
