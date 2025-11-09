import serial
import numpy as np
import time

class LEDController:
    def __init__(self, ports):
        """
        Initialize connections to multiple Arduinos
        ports: list of serial port names like ['/dev/ttyACM0', '/dev/ttyACM1', ...]
        """
        self.arduinos = []
        
        for port in ports:
            try:
                arduino = serial.Serial(port, 9600, timeout=1)
                time.sleep(2)
                self.arduinos.append(arduino)
                print(f"Connected to Arduino on {port}")
            except Exception as e:
                print(f"Failed to connect to {port}: {e}")
        
        if len(self.arduinos) == 0:
            raise Exception("No Arduinos connected!")
    
    def send_layer_to_arduino(self, arduino_index, activation_values, delay=0.3):
        """
        Send activation values to specific Arduino
        arduino_index: which Arduino (0, 1, 2, 3...)
        activation_values: numpy array of neuron activations (0-1 range)
        """
        if arduino_index >= len(self.arduinos):
            print(f"Arduino index {arduino_index} out of range")
            return
        
        brightness_values = np.clip(activation_values * 255, 0, 255).astype(int)
        brightness_values = brightness_values[:10]
        
        command = "LAYER:" + ",".join(map(str, brightness_values)) + "\n"
        self.arduinos[arduino_index].write(command.encode())
        
        time.sleep(delay)
    
    def clear_all_arduinos(self):
        """Turn off all LEDs on all Arduinos"""
        for arduino in self.arduinos:
            arduino.write(b"OFF\n")
        time.sleep(0.1)
    
    def close(self):
        """Close all serial connections"""
        for arduino in self.arduinos:
            arduino.close()
