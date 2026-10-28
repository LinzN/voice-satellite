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

import de.linzn.simplyConfiguration.FileConfiguration;

import java.security.SecureRandom;
import java.util.HexFormat;

/**
 * Settings of the plugin, read from its config file with defaults written back on first start.
 */
public final class SatelliteConfig {

    public final String bindAddress;
    public final int port;
    public final String deviceKey;
    public final boolean onlyPrivateNetworks;
    public final int maxDevices;

    /* speech detection */
    public final int endSilenceMs;
    public final int noSpeechTimeoutMs;
    public final int maxListenMs;
    public final double speechFactor;
    public final double minSpeechLevel;

    public final boolean followUpOnQuestion;
    public final int speechChunkChars;

    public final boolean generatedKey;

    public SatelliteConfig(FileConfiguration config) {
        this.bindAddress = string(config, "server.bindAddress", "0.0.0.0");
        this.port = integer(config, "server.port", 3100);
        this.onlyPrivateNetworks = bool(config, "server.onlyPrivateNetworks", true);
        this.maxDevices = integer(config, "server.maxDevices", 16);

        String key = string(config, "server.deviceKey", "");
        boolean generated = false;
        if (key.isBlank() || key.equals("change-me")) {
            byte[] random = new byte[24];
            new SecureRandom().nextBytes(random);
            key = HexFormat.of().formatHex(random);
            config.set("server.deviceKey", key);
            generated = true;
        }
        this.deviceKey = key;
        this.generatedKey = generated;

        this.endSilenceMs = integer(config, "speech.endSilenceMs", 900);
        this.noSpeechTimeoutMs = integer(config, "speech.noSpeechTimeoutMs", 6000);
        this.maxListenMs = integer(config, "speech.maxListenMs", 15000);
        this.speechFactor = number(config, "speech.speechFactor", 3.0);
        this.minSpeechLevel = number(config, "speech.minSpeechLevel", 400);

        this.followUpOnQuestion = bool(config, "conversation.followUpOnQuestion", true);
        this.speechChunkChars = integer(config, "conversation.speechChunkChars", 300);
        config.save();
    }

    static String string(FileConfiguration config, String path, String def) {
        Object value = config.get(path, def);
        return value == null ? def : String.valueOf(value);
    }

    static int integer(FileConfiguration config, String path, int def) {
        Object value = config.get(path, def);
        if (value instanceof Number number) {
            return number.intValue();
        }
        try {
            return Integer.parseInt(String.valueOf(value).trim());
        } catch (NumberFormatException e) {
            return def;
        }
    }

    static double number(FileConfiguration config, String path, double def) {
        Object value = config.get(path, def);
        if (value instanceof Number number) {
            return number.doubleValue();
        }
        try {
            return Double.parseDouble(String.valueOf(value).trim());
        } catch (NumberFormatException e) {
            return def;
        }
    }

    static boolean bool(FileConfiguration config, String path, boolean def) {
        Object value = config.get(path, def);
        if (value instanceof Boolean b) {
            return b;
        }
        return value == null ? def : Boolean.parseBoolean(String.valueOf(value).trim());
    }

}
