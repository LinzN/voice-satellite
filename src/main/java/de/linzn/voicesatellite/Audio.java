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

import java.io.ByteArrayOutputStream;

/**
 * PCM helpers: windowed sinc resampling and WAV packing.
 */
public final class Audio {

    private static final int TAPS = 16;

    private Audio() {
    }

    public static short[] toSamples(byte[] pcm) {
        short[] samples = new short[pcm.length / 2];
        for (int i = 0; i < samples.length; i++) {
            samples[i] = (short) ((pcm[2 * i] & 0xFF) | (pcm[2 * i + 1] << 8));
        }
        return samples;
    }

    public static byte[] toBytes(short[] samples) {
        byte[] pcm = new byte[samples.length * 2];
        for (int i = 0; i < samples.length; i++) {
            pcm[2 * i] = (byte) samples[i];
            pcm[2 * i + 1] = (byte) (samples[i] >> 8);
        }
        return pcm;
    }

    /**
     * Change the sample rate with a Hann windowed sinc low pass, good enough for speech.
     */
    public static short[] resample(short[] input, int inRate, int outRate) {
        if (inRate == outRate || input.length == 0) {
            return input.clone();
        }
        double ratio = (double) inRate / outRate;
        /* low pass below the lower Nyquist frequency */
        double cutoff = Math.min(1.0, (double) outRate / inRate) * 0.92;
        int outLength = (int) Math.floor(input.length / ratio);
        short[] output = new short[outLength];
        int halfWidth = (int) Math.ceil(TAPS / cutoff);
        for (int n = 0; n < outLength; n++) {
            double center = n * ratio;
            int first = (int) Math.floor(center) - halfWidth + 1;
            double sum = 0;
            double weights = 0;
            for (int k = first; k < first + 2 * halfWidth; k++) {
                if (k < 0 || k >= input.length) {
                    continue;
                }
                double x = k - center;
                double sinc = x == 0 ? 1.0 : Math.sin(Math.PI * cutoff * x) / (Math.PI * cutoff * x);
                double window = 0.5 + 0.5 * Math.cos(Math.PI * x / halfWidth);
                if (Math.abs(x) >= halfWidth) {
                    window = 0;
                }
                double weight = sinc * window;
                sum += input[k] * weight;
                weights += weight;
            }
            double value = weights == 0 ? 0 : sum / weights;
            output[n] = (short) Math.max(Short.MIN_VALUE, Math.min(Short.MAX_VALUE, Math.round(value)));
        }
        return output;
    }

    /**
     * Wrap PCM 16 bit mono into a WAV file.
     */
    public static byte[] wav(byte[] pcm, int sampleRate) {
        ByteArrayOutputStream out = new ByteArrayOutputStream(pcm.length + 44);
        writeAscii(out, "RIFF");
        writeInt(out, 36 + pcm.length);
        writeAscii(out, "WAVE");
        writeAscii(out, "fmt ");
        writeInt(out, 16);
        writeShort(out, 1);              // PCM
        writeShort(out, 1);              // mono
        writeInt(out, sampleRate);
        writeInt(out, sampleRate * 2);   // byte rate
        writeShort(out, 2);              // block align
        writeShort(out, 16);             // bits per sample
        writeAscii(out, "data");
        writeInt(out, pcm.length);
        out.writeBytes(pcm);
        return out.toByteArray();
    }

    private static void writeAscii(ByteArrayOutputStream out, String text) {
        for (int i = 0; i < text.length(); i++) {
            out.write(text.charAt(i));
        }
    }

    private static void writeInt(ByteArrayOutputStream out, int value) {
        out.write(value);
        out.write(value >> 8);
        out.write(value >> 16);
        out.write(value >> 24);
    }

    private static void writeShort(ByteArrayOutputStream out, int value) {
        out.write(value);
        out.write(value >> 8);
    }
}
