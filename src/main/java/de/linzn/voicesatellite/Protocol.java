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
 * Wire protocol of the satellites, see firmware/mirra-voice/protocol.h.
 * Frame: 1 byte type, 4 bytes payload length (big endian), payload.
 * Audio: PCM 16 bit little endian, mono, 16 kHz.
 */
public final class Protocol {

    public static final int VERSION = 1;
    public static final int SAMPLE_RATE = 16000;
    public static final int MAX_FRAME_PAYLOAD = 65536;

    /* device -> server */
    public static final int HELLO = 0x01;
    public static final int AUDIO = 0x02;
    public static final int EVENT = 0x03;
    public static final int PING = 0x04;

    /* server -> device */
    public static final int CHALLENGE = 0x80;
    public static final int WELCOME = 0x81;
    public static final int PLAY = 0x82;
    public static final int PLAY_END = 0x83;
    public static final int STATE = 0x84;
    public static final int STOP_LISTEN = 0x85;
    public static final int PONG = 0x86;
    public static final int CMD = 0x87;
    public static final int ERROR = 0x88;

    private Protocol() {
    }
}
