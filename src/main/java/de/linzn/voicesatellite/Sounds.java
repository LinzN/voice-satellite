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
import java.time.LocalDate;
import java.time.temporal.ChronoUnit;
import java.util.Arrays;
import java.util.Locale;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Sounds the devices play, as PCM 16 bit mono with {@link Protocol#SAMPLE_RATE}. A WAV file
 * {@code sounds/<name>.wav} in the plugin folder replaces the built-in sound (PCM 16 bit, mono or stereo, any
 * sample rate).
 */
final class Sounds {

    private Sounds() {
    }

    static byte[] load(File folder, String name, LocalDate today) throws IOException {
        File file = choose(folder, name, today);
        if (file != null) {
            return Audio.toBytes(readWav(Files.readAllBytes(file.toPath())));
        }
        if (name.equalsIgnoreCase("doorbell")) {
            return Audio.toBytes(doorbell());
        }
        throw new IllegalArgumentException("Unknown sound " + name + " (no file in " + new File(folder, name).getAbsolutePath() + ")");
    }

    /**
     * The file for a sound today, or null for the built-in one. In {@code sounds/<name>/}:
     * <ul>
     *     <li>an occasion file wins while its days last: {@code halloween.wav} (Oct 29 - Nov 2),
     *     {@code christmas.wav} / {@code weihnachten.wav} (Dec 24 - 26), {@code newyear.wav} / {@code silvester.wav}
     *     (Dec 31 - Jan 1), {@code easter.wav} / {@code ostern.wav} (Good Friday - Easter Monday), or an own range
     *     {@code MM-DD_MM-DD[_text].wav}, e.g. {@code 12-01_12-23_advent.wav}; if several match, the shortest range</li>
     *     <li>otherwise {@code <name>.wav} in that folder, all year</li>
     * </ul>
     * {@code sounds/<name>.wav} directly in the sounds folder still works as all-year file.
     */
    static File choose(File folder, String name, LocalDate today) {
        String clean = name.toLowerCase(Locale.ROOT);
        if (!clean.matches("[a-z0-9_-]{1,32}")) {
            throw new IllegalArgumentException("Invalid sound name " + name);
        }
        File directory = new File(folder, clean);
        File[] files = directory.listFiles((dir, file) -> file.toLowerCase(Locale.ROOT).endsWith(".wav"));
        if (files != null) {
            File best = null;
            long bestDays = Long.MAX_VALUE;
            Arrays.sort(files);
            for (File file : files) {
                String stem = file.getName().substring(0, file.getName().length() - 4).toLowerCase(Locale.ROOT);
                LocalDate[] range = occasion(stem, today);
                if (range != null) {
                    long days = ChronoUnit.DAYS.between(range[0], range[1]);
                    if (days < bestDays && file.isFile()) {
                        best = file;
                        bestDays = days;
                    }
                }
            }
            if (best != null) {
                return best;
            }
            File all = new File(directory, clean + ".wav");
            if (all.isFile()) {
                return all;
            }
        }
        File old = new File(folder, clean + ".wav");
        return old.isFile() ? old : null;
    }

    private static final Pattern RANGE = Pattern.compile("(\\d{2})-(\\d{2})_(\\d{2})-(\\d{2})(_.*)?");

    /**
     * @return first and last day of the occasion that contains today, or null if the name is no occasion or it is
     * not its time
     */
    static LocalDate[] occasion(String stem, LocalDate today) {
        for (int year = today.getYear() - 1; year <= today.getYear() + 1; year++) {
            LocalDate[] range = occasionIn(stem, year);
            if (range == null) {
                return null;
            }
            if (!today.isBefore(range[0]) && !today.isAfter(range[1])) {
                return range;
            }
        }
        return null;
    }

    /**
     * @return the occasion starting in the given year, or null if the name is no occasion
     */
    private static LocalDate[] occasionIn(String stem, int year) {
        switch (stem) {
            case "halloween" -> {
                LocalDate day = LocalDate.of(year, 10, 31);
                return new LocalDate[]{day.minusDays(2), day.plusDays(2)};
            }
            case "christmas", "weihnachten" -> {
                return new LocalDate[]{LocalDate.of(year, 12, 24), LocalDate.of(year, 12, 26)};
            }
            case "newyear", "silvester" -> {
                return new LocalDate[]{LocalDate.of(year, 12, 31), LocalDate.of(year + 1, 1, 1)};
            }
            case "easter", "ostern" -> {
                LocalDate sunday = easterSunday(year);
                return new LocalDate[]{sunday.minusDays(2), sunday.plusDays(1)};
            }
            default -> {
                Matcher matcher = RANGE.matcher(stem);
                if (!matcher.matches()) {
                    return null;
                }
                try {
                    LocalDate from = LocalDate.of(year, Integer.parseInt(matcher.group(1)), Integer.parseInt(matcher.group(2)));
                    LocalDate to = LocalDate.of(year, Integer.parseInt(matcher.group(3)), Integer.parseInt(matcher.group(4)));
                    /* a range over new year, e.g. 12-20_01-06 */
                    return new LocalDate[]{from, to.isBefore(from) ? to.plusYears(1) : to};
                } catch (java.time.DateTimeException e) {
                    return null;
                }
            }
        }
    }

    /**
     * Easter Sunday (Gregorian calendar, anonymous Gregorian algorithm).
     */
    static LocalDate easterSunday(int year) {
        int a = year % 19, b = year / 100, c = year % 100, d = b / 4, e = b % 4, f = (b + 8) / 25, g = (b - f + 1) / 3;
        int h = (19 * a + b - d - g + 15) % 30, i = c / 4, k = c % 4, l = (32 + 2 * e + 2 * i - h - k) % 7;
        int m = (a + 11 * h + 22 * l) / 451;
        int month = (h + l - 7 * m + 114) / 31, day = (h + l - 7 * m + 114) % 31 + 1;
        return LocalDate.of(year, month, day);
    }

    /**
     * Two soft, glass-like chords (E6+G#6, then C#6+F#6), played twice, with a little room echo.
     */
    static short[] doorbell() {
        int rate = Protocol.SAMPLE_RATE;
        double[] mix = new double[(int) (rate * 3.4)];
        for (double repeat : new double[]{0.0, 1.5}) {
            for (double frequency : new double[]{1318.51, 1661.22}) {
                tone(mix, rate, repeat, frequency, 1.0, 0.7);
            }
            for (double frequency : new double[]{1108.73, 1479.98}) {
                tone(mix, rate, repeat + 0.35, frequency, 0.9, 0.8);
            }
        }
        double[] wet = echo(mix, rate, 0.35);
        double peak = 1e-9;
        for (double value : wet) {
            peak = Math.max(peak, Math.abs(value));
        }
        short[] samples = new short[wet.length];
        double gain = 0.8 * 32767 / peak;
        for (int i = 0; i < wet.length; i++) {
            samples[i] = (short) Math.round(wet[i] * gain);
        }
        return samples;
    }

    /**
     * FM tone with the modulator at 3.5 times the frequency and a quickly decaying index: a glassy, bell-like strike.
     */
    private static void tone(double[] mix, int rate, double start, double frequency, double amplitude, double decay) {
        int first = (int) (start * rate);
        int length = Math.min((int) (1.8 * rate), mix.length - first);
        for (int i = 0; i < length; i++) {
            double t = (double) i / rate;
            double index = 0.9 * Math.exp(-t / 0.05);
            double envelope = amplitude * Math.exp(-t / decay) * Math.min(1.0, t / 0.003);
            mix[first + i] += envelope * Math.sin(2 * Math.PI * frequency * t + index * Math.sin(2 * Math.PI * frequency * 3.5 * t));
        }
    }

    /**
     * Four feedback delays: a small, bright room.
     */
    private static double[] echo(double[] dry, int rate, double amount) {
        double[] out = dry.clone();
        double[][] lines = {{0.029, 0.5}, {0.037, 0.45}, {0.041, 0.4}, {0.053, 0.35}};
        for (double[] line : lines) {
            int delay = (int) (line[0] * rate);
            double[] comb = new double[dry.length];
            for (int i = delay; i < dry.length; i++) {
                comb[i] = dry[i - delay] + line[1] * comb[i - delay];
            }
            for (int i = 0; i < dry.length; i++) {
                out[i] += amount * comb[i] / lines.length;
            }
        }
        return out;
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
