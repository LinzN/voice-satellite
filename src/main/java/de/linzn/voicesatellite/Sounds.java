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

import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.util.Locale;

/**
 * Sounds the devices play, as PCM 16 bit mono with {@link Protocol#SAMPLE_RATE}. A WAV file
 * {@code sounds/<name>.wav} in the plugin folder replaces the built-in sound (PCM 16 bit, mono or stereo, any
 * sample rate).
 */
final class Sounds {

    private Sounds() {
    }

    static byte[] load(File folder, String name) throws IOException {
        String clean = name.toLowerCase(Locale.ROOT);
        if (!clean.matches("[a-z0-9_-]{1,32}")) {
            throw new IllegalArgumentException("Invalid sound name " + name);
        }
        File custom = new File(folder, clean + ".wav");
        if (custom.isFile()) {
            return Audio.toBytes(readWav(Files.readAllBytes(custom.toPath())));
        }
        if (clean.equals("doorbell")) {
            return Audio.toBytes(doorbell());
        }
        throw new IllegalArgumentException("Unknown sound " + name + " (no " + custom.getAbsolutePath() + ")");
    }

    /**
     * Two struck chime bars, high then low ("ding-dong"), played twice.
     */
    static short[] doorbell() {
        int rate = Protocol.SAMPLE_RATE;
        double[] mix = new double[(int) (rate * 3.6)];
        double[][] strikes = {{0.00, 659.25}, {0.55, 523.25}, {1.65, 659.25}, {2.20, 523.25}};
        for (double[] strike : strikes) {
            strike(mix, rate, strike[0], strike[1]);
        }
        double peak = 1e-9;
        for (double value : mix) {
            peak = Math.max(peak, Math.abs(value));
        }
        short[] samples = new short[mix.length];
        double gain = 0.75 * 32767 / peak;
        for (int i = 0; i < mix.length; i++) {
            samples[i] = (short) Math.round(mix[i] * gain);
        }
        return samples;
    }

    private static void strike(double[] mix, int rate, double start, double frequency) {
        /* partial frequency ratio, amplitude, decay time in seconds: a bar sounds mostly its fundamental */
        double[][] partials = {{1.0, 1.0, 0.9}, {2.0, 0.35, 0.45}, {3.0, 0.12, 0.25}, {4.16, 0.06, 0.12}};
        int first = (int) (start * rate);
        for (int i = first; i < mix.length; i++) {
            double t = (double) (i - first) / rate;
            double attack = Math.min(1.0, t / 0.004);
            double value = 0;
            for (double[] partial : partials) {
                value += partial[1] * Math.exp(-t / partial[2]) * Math.sin(2 * Math.PI * frequency * partial[0] * t);
            }
            mix[i] += attack * value;
        }
    }

    /**
     * @return mono samples at {@link Protocol#SAMPLE_RATE}
     */
    static short[] readWav(byte[] file) throws IOException {
        ByteBuffer buffer = ByteBuffer.wrap(file).order(ByteOrder.LITTLE_ENDIAN);
        if (file.length < 12 || buffer.getInt(0) != 0x46464952 || buffer.getInt(8) != 0x45564157) {
            throw new IOException("Not a WAV file");
        }
        int channels = 0, rate = 0, bits = 0, format = 0;
        int position = 12;
        while (position + 8 <= file.length) {
            int id = buffer.getInt(position);
            int size = buffer.getInt(position + 4);
            int body = position + 8;
            if (size < 0 || body + size > file.length) {
                size = file.length - body;
            }
            if (id == 0x20746D66) { // "fmt "
                format = buffer.getShort(body) & 0xFFFF;
                channels = buffer.getShort(body + 2);
                rate = buffer.getInt(body + 4);
                bits = buffer.getShort(body + 14);
            } else if (id == 0x61746164) { // "data"
                if ((format != 1 && format != 0xFFFE) || bits != 16 || channels < 1 || channels > 2 || rate <= 0) {
                    throw new IOException("Only PCM 16 bit mono or stereo WAV files");
                }
                int frames = size / (2 * channels);
                short[] mono = new short[frames];
                for (int i = 0; i < frames; i++) {
                    int sum = 0;
                    for (int c = 0; c < channels; c++) {
                        sum += buffer.getShort(body + (i * channels + c) * 2);
                    }
                    mono[i] = (short) (sum / channels);
                }
                return Audio.resample(mono, rate, Protocol.SAMPLE_RATE);
            }
            position = body + size + (size & 1);
        }
        throw new IOException("WAV file without audio data");
    }
}
