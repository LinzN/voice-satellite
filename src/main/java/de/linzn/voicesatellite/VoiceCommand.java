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

import de.linzn.mirra.voice.MirraVoice;
import de.linzn.stem.modules.commandModule.ICommand;
import org.json.JSONObject;

import java.util.Arrays;

/**
 * STEM command: voice list | voice say &lt;device|all&gt; &lt;text&gt; | voice volume &lt;device&gt; &lt;0-10&gt; | voice reboot &lt;device&gt;
 * | voice record &lt;device|all&gt; on|off (save recordings only, for wake word training)
 * | voice capture &lt;device|all&gt; on|off (save recordings and answer normally, to check the microphone)
 */
public class VoiceCommand implements ICommand {

    private final VoiceSatellitePlugin plugin;

    public VoiceCommand(VoiceSatellitePlugin plugin) {
        this.plugin = plugin;
    }

    @Override
    public boolean executeTerminal(String[] args) {
        if (args.length == 0 || args[0].equalsIgnoreCase("list")) {
            for (MirraVoice.Device device : this.plugin.devices()) {
                String extra = this.plugin.connection(device.id()).map(c -> " fw " + c.firmware() + " @ " + c.remoteAddress()).orElse("");
                this.plugin.getPluginLogger().info(device.id() + " (" + device.name() + ") " + (device.online() ? "online" : "offline") + extra);
            }
            return true;
        }
        try {
            switch (args[0].toLowerCase()) {
                case "say" -> {
                    if (args.length < 3) {
                        return false;
                    }
                    String text = String.join(" ", Arrays.copyOfRange(args, 2, args.length));
                    if (args[1].equalsIgnoreCase("all")) {
                        for (SatelliteConnection connection : this.plugin.connections()) {
                            this.plugin.getMirraVoice().speak(connection.deviceId(), text);
                        }
                    } else {
                        this.plugin.getMirraVoice().speak(this.plugin.connection(args[1]).map(SatelliteConnection::deviceId).orElse(args[1]), text);
                    }
                }
                case "volume" -> {
                    if (args.length < 3) {
                        return false;
                    }
                    this.plugin.sendCommand(args[1], new JSONObject().put("cmd", "volume").put("value", Integer.parseInt(args[2])));
                }
                case "record", "capture" -> {
                    if (args.length < 3) {
                        return false;
                    }
                    boolean on = args[2].equalsIgnoreCase("on");
                    VoiceSatellitePlugin.CaptureMode mode = !on ? VoiceSatellitePlugin.CaptureMode.OFF
                            : args[0].equalsIgnoreCase("record") ? VoiceSatellitePlugin.CaptureMode.RECORD
                            : VoiceSatellitePlugin.CaptureMode.CAPTURE;
                    this.plugin.setCaptureMode(args[1], mode);
                    this.plugin.getPluginLogger().info("Capture mode of " + args[1] + ": " + mode + ", files in "
                            + this.plugin.recordingsFolder().getAbsolutePath());
                }
                case "reboot" -> {
                    if (args.length < 2) {
                        return false;
                    }
                    this.plugin.sendCommand(args[1], new JSONObject().put("cmd", "reboot"));
                }
                default -> {
                    return false;
                }
            }
            this.plugin.getPluginLogger().info("Done");
        } catch (Exception e) {
            this.plugin.getPluginLogger().warning("voice " + args[0] + " failed: " + e.getMessage());
        }
        return true;
    }
}
