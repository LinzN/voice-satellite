/*
 * Copyright (c) 2026 MirraNET, Niklas Linz. All rights reserved.
 *
 * This file is part of the MirraNET project and is licensed under the
 * GNU Lesser General Public License v3.0 (LGPLv3).
 *
 * You may use, distribute and modify this code under the terms
 * of the LGPLv3 license. You should have received a copy of the
 * license along with this file. If not, see <https://www.gnu.org/licenses/lgpl-3.0.html>
 * or contact: niklas.linz@mirranet.de
 */


package de.linzn.voicesatellite;

/**
 * Energy based end of speech detection. The background level is the quietest recent frame (it rises slowly),
 * so it also works when the person is already talking when the recording starts (after the wake word).
 * Speech starts when the level rises clearly above the background, and the recording ends after a pause of
 * {@code endSilenceMs}.
 */
public final class SpeechDetector {

    public enum Result { CONTINUE, END_OF_SPEECH, NO_SPEECH, TOO_LONG }

    private static final int START_SPEECH_MS = 64;
    private static final double NOISE_RISE = 0.003;

    private final SatelliteConfig config;
    private double noise = -1;
    private long totalSamples;
    private long loudSamples;
    private long silentSamples;
    private boolean speech;

    public SpeechDetector(SatelliteConfig config) {
        this.config = config;
    }

    public boolean heardSpeech() {
        return this.speech;
    }

    public long durationMs() {
        return this.totalSamples * 1000 / Protocol.SAMPLE_RATE;
    }

    /**
     * Feed one frame of 16 bit samples.
     */
    public Result feed(short[] samples, int count) {
        if (count <= 0) {
            return Result.CONTINUE;
        }
        double sum = 0;
        for (int i = 0; i < count; i++) {
            sum += (double) samples[i] * samples[i];
        }
        double rms = Math.sqrt(sum / count);
        this.totalSamples += count;

        if (this.noise < 0 || rms < this.noise) {
            this.noise = rms;
        } else {
            this.noise += (rms - this.noise) * NOISE_RISE;
        }

        double threshold = Math.max(this.config.minSpeechLevel, this.noise * this.config.speechFactor);
        if (rms > threshold) {
            this.loudSamples += count;
            this.silentSamples = 0;
            if (!this.speech && this.loudSamples * 1000 / Protocol.SAMPLE_RATE >= START_SPEECH_MS) {
                this.speech = true;
            }
        } else {
            if (!this.speech) {
                this.loudSamples = 0;
            }
            if (rms < threshold * 0.7) {
                this.silentSamples += count;
            }
        }

        long elapsedMs = this.durationMs();
        if (elapsedMs >= this.config.maxListenMs) {
            return Result.TOO_LONG;
        }
        if (!this.speech && elapsedMs >= this.config.noSpeechTimeoutMs) {
            return Result.NO_SPEECH;
        }
        if (this.speech && this.silentSamples * 1000 / Protocol.SAMPLE_RATE >= this.config.endSilenceMs) {
            return Result.END_OF_SPEECH;
        }
        return Result.CONTINUE;
    }
}
