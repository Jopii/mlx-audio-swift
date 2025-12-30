//
//  MLXAudioTests.swift
//  MLXAudioTests
//
//  Created by Ben Harraway on 14/04/2025.
//

import Testing
import MLX
import Foundation

@testable import MLXAudioTTS
@testable import MLXAudioCodecs

struct SNACTests {

    @Test func testSNACEncodeDecodeCycle() async throws {
        // 1. Load audio from file
        let audioURL = URL(fileURLWithPath: "/Users/prince_canuma/taycan.wav")
        let (sampleRate, audioData) = try loadAudioArray(from: audioURL)
        print("Loaded audio: \(audioData.shape), sample rate: \(sampleRate)")

        // 2. Load SNAC model from HuggingFace (24kHz model)
        print("Loading SNAC model...")
        let snac = try await SNAC.fromPretrained("mlx-community/snac_24khz")
        print("SNAC model loaded!")

        // 3. Reshape audio for SNAC: [batch, channels, samples]
        let audioInput = audioData.reshaped([1, 1, audioData.shape[0]])
        print("Audio input shape: \(audioInput.shape)")

        // 4. Encode audio to codes
        print("Encoding audio...")
        let codes = snac.encode(audioInput)
        print("Encoded to \(codes.count) codebook levels:")
        for (i, code) in codes.enumerated() {
            print("  Level \(i): \(code.shape)")
        }

        // 5. Decode codes back to audio
        print("Decoding audio...")
        let reconstructed = snac.decode(codes)
        print("Reconstructed audio shape: \(reconstructed.shape)")

        // 6. Save reconstructed audio
        let outputURL = URL(fileURLWithPath: "/Users/prince_canuma/taycan_reconstructed.wav")
        let outputAudio = reconstructed.squeezed()  // Remove batch/channel dims
        try saveAudioArray(outputAudio, sampleRate: Double(snac.samplingRate), to: outputURL)
        print("Saved reconstructed audio to: \(outputURL.path)")

        // Basic check: output should have samples
        #expect(reconstructed.shape.last! > 0)
    }
}

//struct MLXAudioTests {
//
//    func example() async throws {
//        // Write your test here and use APIs like `#expect(...)` to check expected conditions.
//    }
//
//    func testViewBodyDoesNotCrash() {
//        _ = ContentView().body
//    }
//
//    func testKokoro() async {
//        let kokoroTTSModel = KokoroTTSModel()
//        kokoroTTSModel.say("test", .bmGeorge)
//    }
//
//    func testOrpheus() async {
//        let orpheusTTSModel = OrpheusTTSModel()
//        await orpheusTTSModel.say("test", .tara)
//    }
//}
