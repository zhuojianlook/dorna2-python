#!/usr/bin/env python3
"""
50 0 Hz Xbox Trigger → Velocity Bridge

• RT (axis 5) → CW velocity
• LT (axis 2) → CCW velocity
• X  (btn 2)  → reset/stop
• Sends V<rate>\n and R\n to the Arduino
• Drains any incoming serial data after each send
"""

import time
import pygame
import serial

# ───── Config ─────
SERIAL_PORT    = '/dev/serial/by-id/usb-Arduino__www.arduino.cc__0043_44236313735351100201-if00'
BAUDRATE       = 115200
MAX_RATE       = 800     # steps/sec at full trigger
DEADZONE       = 0.02
RIGHT_AXIS     = 5
LEFT_AXIS      = 2
RESET_BUTTON   = 2
LOOP_HZ        = 500
LOOP_DT        = 1.0 / LOOP_HZ

def normalize(v):
    # normalize [-1..1] to [0..1]
    return (v + 1.0) / 2.0 if v < -0.2 or v > 1.0 else v

def main():
    # Open serial port
    ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=0.1)
    time.sleep(2)  # let Arduino reset
    print(f"[Bridge] Serial open {SERIAL_PORT} @ {BAUDRATE}")

    # Init joystick
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        print("[Bridge] No joystick detected")
        return
    js = pygame.joystick.Joystick(0)
    js.init()
    print(f"[Bridge] Using joystick: {js.get_name()}")
    print("Ready: RT→CW, LT→CCW, X→reset")

    last_rate = None
    clock = pygame.time.Clock()

    try:
        while True:
            pygame.event.pump()

            # read and normalize triggers
            rt = normalize(js.get_axis(RIGHT_AXIS))
            lt = normalize(js.get_axis(LEFT_AXIS))

            # determine signed rate
            if rt > DEADZONE:
                rate = int(rt * MAX_RATE)
            elif lt > DEADZONE:
                rate = -int(lt * MAX_RATE)
            else:
                rate = 0

            # send V<rate> only on change
            if rate != last_rate:
                cmd = f"V{rate}\n".encode()
                ser.write(cmd)
                last_rate = rate
                # drain any Arduino output so buffer stays clear
                if ser.in_waiting:
                    ser.read(ser.in_waiting)

            # reset on X button
            if js.get_button(RESET_BUTTON):
                ser.write(b"R\n")
                # drain
                if ser.in_waiting:
                    ser.read(ser.in_waiting)
                last_rate = 0
                time.sleep(0.1)  # debounce

            clock.tick_busy_loop(LOOP_HZ)

    except KeyboardInterrupt:
        print("\n[Bridge] Interrupted, exiting.")
    finally:
        ser.close()
        print("[Bridge] Serial port closed.")

if __name__ == "__main__":
    main()
