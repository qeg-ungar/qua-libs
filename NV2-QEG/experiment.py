# general python imports
import numpy as np

# user defined classes
from config import u
from base_experiment import Experiment


class NVExperiment(Experiment):

    def setup_cw_odmr(self, f_vec, readout_len=None, wait_time=1_000, amplitude=1):
        """
        Sequence of commands to run a continuous wave ODMR experiment.

        Args:
            f_vec (array): Array of frequencies to sweep over
            readout_len (int): time of measurement acquisition. Defaults to the config's `long_meas_len_1`
            wait_time (int, optional): Wait time after CW before readout. Should exceed metastable state lifetime.
                Defaults to 1_000.
            amplitude (int, optional): Amplitude of the microwave drive. Defaults to 1.
        """
        readout_len = readout_len if readout_len is not None else self.config.long_meas_len_1

        self.add_align()
        self.add_frequency_update("NV", f_vec)

        self.add_laser(channel="AOM1", length=readout_len)
        self.add_cw_drive("NV", readout_len, amplitude)

        self.add_wait(wait_time)
        self.add_measure(channel="SPCM1", mode="long_readout", meas_len=readout_len)
        self.add_measure_delay(1_000)

        # for plotting results
        self.x_axis_scale = 1 / u.MHz
        self.x_axis_label = "MW frequency [MHz]"
        self.plot_title = "CW ODMR"

    # def setup_time_rabi(self, t_vec=np.arange(16, 200, 4)):
    #     """
    #     Sequence of commands to run a Rabi experiment sweeping time of MW.

    #     Args:
    #         t_vec (array): Array of pulse durations in clock cycles (4ns)
    #     """

    #     self.add_initialization(channel="AOM1")
    #     self.add_pulse("x180", "NV", amplitude=1, length=t_vec)
    #     self.add_align()
    #     self.add_laser(mode="laser_ON", channel="AOM1")
    #     self.add_measure(channel="SPCM1")

    #     # for plotting results
    #     self.file_prefix = "time_rabi"
    #     self.x_axis_scale = 4
    #     self.x_axis_label = "Rabi pulse duration [ns]"
    #     self.plot_title = "Time Rabi"

    # def setup_power_rabi(self, a_vec=np.arange(0.1, 2, 0.02)):
    #     """
    #     Sequence of commands to run a Rabi experiment sweeping time of MW.

    #     Args:
    #         a_vec (array): Array of pulse voltage scalings in [a.u.]
    #     """

    #     self.add_initialization(channel="AOM1")
    #     self.add_pulse("x180", "NV", amplitude=a_vec, length=self.config.x180_len_NV)
    #     self.add_align()
    #     self.add_laser(mode="laser_ON", channel="AOM1")
    #     self.add_measure(channel="SPCM1")

    #     # for plotting results
    #     self.x_axis_scale = self.config.x180_amp_NV
    #     self.x_axis_label = "Rabi pulse amplitude [V]"
    #     self.plot_title = "Power Rabi"

    # def setup_time_rabi(self, t_vec=np.arange(4, 40, 4)):
    #     """
    #     Sequence of commands to run a Rabi experiment sweeping time of MW.

    #     Args:
    #         t_vec (array): Array of pulse durations in ns (integer multiples of 4ns)
    #     """

    #     self.rabi_sequence(length=t_vec)

    #     # for plotting results
    #     self.x_axis_scale = 4
    #     self.x_axis_label = "Rabi pulse duration [ns]"
    #     self.plot_title = "Time Rabi"

    def setup_power_rabi(self, a_vec=np.arange(0.1, 2, 0.02)):
        """
        Sequence of commands to run a Rabi experiment sweeping amplitude of MW pulse.

        Args:
            a_vec (array): Array of pulse voltage scalings in [a.u.]
        """

        self.rabi_sequence(amplitude=a_vec)

        # for plotting results
        self.x_axis_scale = self.config.x180_amp_NV
        self.x_axis_label = "Rabi pulse amplitude [V]"
        self.plot_title = "Power Rabi"

    def setup_pulsed_odmr(self, f_vec=np.arange(60, 100, 1) * u.MHz, amplitude=1):
        """
        Sequence of commands to run a Rabi experiment sweeping time of MW.

        Args:
            t_vec (array): Array of pulse durations in clock cycles (4ns)
        """

        self.rabi_sequence(frequency=f_vec, amplitude=amplitude)

        # for plotting results
        self.x_axis_scale = 1 / u.MHz
        self.x_axis_label = "MW frequency [MHz]"
        self.plot_title = "Pulsed ODMR"

    def rabi_sequence(self, frequency=None, amplitude=1, length=None):

        if frequency is not None:
            self.add_frequency_update("NV", frequency)
        length = length if length is not None else self.config.x180_len_NV
        self.add_initialization(channel="AOM1")
        self.add_pulse("x180", "NV", amplitude=amplitude, length=length)
        self.add_align()
        self.add_laser(mode="laser_ON", channel="AOM1")
        self.add_measure(channel="SPCM1")
