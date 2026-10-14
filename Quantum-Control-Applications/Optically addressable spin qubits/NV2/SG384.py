import numpy as np
import pyvisa
import time


class SG384Control:
    def __init__(self, visa_address):
        self.visa_address = visa_address
        self.instr = None
        self.connect()

    def connect(self):
        rm = pyvisa.ResourceManager()
        self.instr = rm.open_resource(self.visa_address)
        self.instr.read_termination = "\n"
        self.instr.write_termination = "\n"
        time.sleep(0.1)
        if self.instr.query("*IDN?"):
            print(f"Connected to SG384 at {self.visa_address}")
        else:
            print("ERROR! CALL THE COMPUTER POLICE")

    def send_command(self, command):
        r"""
        Writes command to the device and waits 100ms. No error handling enabled.
        """
        self.instr.write(command)
        time.sleep(0.1)  # Add a small delay to ensure the command is processed

    def set_amplitude(self, amplitude):
        r"""
        Sets the amplitude of the type-N RF output to `amplitude` in units of dBm
        """
        command = f"AMPR {amplitude}"
        self.send_command(command)
        # print(f"Set amplitude to {amplitude}")

    def set_amplitude_lf(self, amplitude):
        r"""
        Sets the amplitude of the BNC LF output to `amplitude` in units of dBm
        """
        command = f"AMPL {amplitude}"
        self.send_command(command)
        # print(f"Set amplitude to {amplitude}")

    def set_frequency(self, frequency):
        r"""
        Sets the frequency of the SG384 to `frequency` in units of Hz.
        """
        command = f"FREQ {frequency}"
        self.send_command(command)
        # print(f"Set frequency to {frequency/1e6} MHz")

    def get_frequency(self):
        r"""
        Get the frequency of the SG384 device, in units of MHz
        """
        return float(self.instr.query("FREQ?MHz"))

    def rf_on(self):
        self.ntype_on(True)

    def rf_off(self):
        self.ntype_off(False)

    def ntype_on(self, boolean,print_me=True):
        r"""
        Turns on/off the rf. send `boolean`=True or 1 to turn on RF. Send `boolean`=False or 0
        to turn off the rf.
        """
        command = f"ENBR {boolean}"
        self.send_command(command)
        if print_me:
            print(f"N-type RF bool set to {boolean}")

    def bnctype_on(self, bool2,print_me = True):
        r"""
        """
        command = f"ENBL {bool2}"
        self.send_command(command)
        if print_me:
            print(f"bnc-type RF bool set to {bool2}")

    def set_modulation_state(self, status):
        """
        Set the status of the modulation

        Input:
            status (string) : 'On' or 'Off'

        Output:
            None
        """
        if status.upper() == "ON":
            status = 1
        elif status.upper() == "OFF":
            status = 0
        else:
            raise ValueError("set_status(): can only set on or off")
        self.send_command("MODL%s" % status)

    def set_modulation_type(self, mtype):

        type_dict = {"AM": 0, "FM": 1, "PHASEM": 2, "SWEEP": 3, "PULSE": 4, "BLANK": 5, "IQ": 6}
        self.send_command("TYPE %s" % type_dict[mtype])

    def close(self):
        self.instr.close()
        print("Connection closed.")
