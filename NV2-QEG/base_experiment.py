# general python imports
from datetime import datetime
import json
import numpy as np
from collections.abc import Iterable
import matplotlib
import matplotlib.pyplot as plt

# user defined classes
from utils import NumpyEncoder
from config import ConfigNV, u

# Quantum machines imports
from qm import SimulationConfig
from qm.qua import (
    update_frequency,
    play,
    wait,
    measure,
    save,
    align,
    amp,
    program,
    declare,
    time_tagging,
    stream_processing,
    declare_stream,
    for_,
    fixed,
)
from qualang_tools.plot import interrupt_on_close
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array

# plotting
matplotlib.use("TkAgg")


class Experiment:
    def __init__(self, config=None):
        # containers for commands
        self.var_vec = None
        self.commands = []
        self.use_fixed = False
        self.measure_len = None
        self.measure_mode = None
        self.measure_channel = None
        self.initialize = False
        self.measure_delay = 0
        self.laser_channel = None

        # containers for results
        self.counts0 = None
        self.counts_ref0 = None
        self.counts1 = None
        self.counts_ref1 = None
        self.iteration = None

        # containers for plotting data
        self.x_axis_scale = 1
        self.x_axis_label = "Swept Variable [a.u.]"
        self.y_axis_label = "Intensity [kcps]"
        self.plot_title = "Measurement Results"

        # store the config
        self.config = config if config is not None else ConfigNV()

        # saving defaults
        self.file_prefix = "expt"

    def add_pulse(self, name, element, amplitude=1, length=None, cycle=False):
        """
        Adds a type "microwave" command to the experiment on the
        desired `element`.

        Args:
            name (string): Name of the pulse. 8 predefined pulses are avaialble,
                "+/-" * "x/y" * "90/180", eg "y180" or "-x90"
            element (string): Channel to play the pulse on, like "NV" or "C13", must be defined in the config
            amplitude (float, Iterable): amplitude of pulse, defaults to 1
            length (int, Iterable): time of the pulse, default to config's `x180_len_NV`, in ns
            cycle (bool): If True, the pulse is inverted during the second cycle performing non-contrast measurements.
        """
        length = length if length is not None else self.config.x180_len_NV
        command = {
            "type": "pulse",
            "element": element,
            "name": name,
            "cycle": cycle,
        }
        if isinstance(amplitude, Iterable):
            command["scale"] = self.update_loop(amplitude)
            self.use_fixed = True
            command["length"] = length // 4
        elif isinstance(length, Iterable):
            command["scale"] = self.update_loop(np.array(length) // 4)
            command["amplitude"] = amplitude
        else:
            command["length"] = length // 4
            command["amplitude"] = amplitude
        self.commands.append(command)

    def add_cw_drive(self, element, length, amplitude=1):
        """
        Adds a type "microwave" command to the experiment on the
        desired `element`.

        Args:
            element (string): Channel to play the pulse on, like "NV" or "C13" in the config
            length (int, Iterable): time of pulse
            amplitude (float, Iterable): amplitude of pulse, default to 1
        """
        command = {"type": "cw", "element": element}
        if isinstance(amplitude, Iterable):
            command["scale"] = self.update_loop(amplitude)
            self.use_fixed = True
            command["length"] = length // 4
        elif isinstance(length, Iterable):
            command["scale"] = self.update_loop(np.array(length) // 4)
            command["amplitude"] = amplitude
        else:
            command["length"] = length // 4
            command["amplitude"] = amplitude
        self.commands.append(command)

    def add_measure_delay(self, length=None):
        """
        Adds a type "measure_delay" command to the experiment.

        Args:
            length (int): time of measurement acquisition in ns, defaults to the config's `meas_len_1`
        """
        self.measure_delay = length if length is not None else self.config.meas_len_1
        self.measure_delay = self.measure_delay // 4

    def add_laser(self, mode="laser_ON", channel="AOM1", length=None):
        """
        Adds a type "laser" command to the experiment

        Args:
            mode (string): Mode of the laser, like "laser_ON" or "laser_OFF"
            channel (string): Channel to play the laser on, like "AOM1" in the config
            length (int): time of the laser pulse in ns. Defaults to the config's `initialization_len_1`
        """
        length = length if length is not None else self.config.initialization_len_1
        length = length // 4
        self.commands.append({"type": "laser", "mode": mode, "channel": channel, "length": length})

        self.laser_channel = channel

    def add_align(self):
        """
        Adds a type "align" command to the experiment.
        """
        self.commands.append({"type": "align"})

    def add_wait(self, length):
        """
        Adds a type "wait" command to the experiment.

        Args:
            length (int, Iterable): Time to wait in ns, or an array of wait-times in ns to define a loop
        """
        if isinstance(length, Iterable):
            scale = self.update_loop(np.array(length) // 4)
            self.commands.append({"type": "wait", "scale": scale})
        else:
            self.commands.append({"type": "wait", "length": length // 4})

    def add_measure(self, mode="readout", channel="SPCM1", meas_len=None):
        """
        Adds a type "measure" command to the experiment.

        Args:
            mode (string): Measurement mode, like "readout" or "long_readout"
            channel (string): Channel to measure on, like "SPCM1" in the config
            meas_len (int): Time of measurement acquisition in ns. Defaults to the config's `meas_len_1`
        """
        meas_len = meas_len if meas_len is not None else self.config.meas_len_1
        meas_len = meas_len // 4
        self.commands.append({"type": "measure", "channel": channel, "mode": mode, "meas_len": meas_len})

        if self.measure_len is None:
            self.measure_len = meas_len
            self.measure_mode = mode
            self.measure_channel = channel
        elif self.measure_len != meas_len:
            raise ValueError("Inconsistent measurement lengths.")

    def add_frequency_update(self, element, freq_list):
        """
        Adds a type "update_frequency" command to the experiment.

        Args:
            element (string): Name of the element to update the frequency of
            freq_list (Iterable): Array of frequencies to update the element to, in Hz
        """
        self.commands.append({"type": "update_frequency", "element": element})
        self.update_loop(freq_list)

    def update_loop(self, var_vec):
        """
        Updates the variable vector for the experiment. This is used to define the loop
        that the experiment will run over. If the variable vector is already defined, this
        function will check that the new vector is consistent with the previous one by determining
        if the new vector is a constant multiple of the old one.

        For internal use only - will have dramatic mutation side effects otherwise.

        Args:
            var_vec (array): Array of values for the variable in the experiment

        Returns:
            float: The constant multiple of the new vector to the old vector, 1 if this is the first update.

        Raises:
            ValueError: Throws an error if the new vector is not a constant multiple of the old one, or if
                the new vector is all zeros.
        """
        if np.all(var_vec == 0):
            raise ValueError("Variable vector cannot be all zeros.")

        if self.var_vec is None:
            self.var_vec = var_vec
            return 1

        two = self.var_vec
        if np.dot(var_vec, two) * np.dot(two, var_vec) == np.dot(var_vec, var_vec) * np.dot(two, two):
            div = -1
            idx = 0
            while div < 0:
                div = two[idx] / var_vec[idx] if var_vec[idx] != 0 else -1
                idx += 1
            if div > 0:
                return div

        raise ValueError("Inconsistent loop variables.")

    def add_initialization(self, channel="AOM1"):
        """
        Adds a laser pulse to polarize the system before the first sequence. This is controlled with the config file.
        """
        self.initialize = True
        self.laser_channel = channel

    def _translate_command(self, command, var, times, counts, counts_st, invert):
        """
        Helper function whcih translates a command dictionary into a QUA command. Plays qua commands, can only
        be called from within a qua program.

        Args:
            command (dict): Command dictionary

        Returns:
            qua command: The QUA command
        """
        scale = command.get("scale", 1)
        svar = (scale * var) if scale != 1 else var
        match command["type"]:
            case "update_frequency":
                update_frequency(command["element"], svar)

            case "pulse":
                amplitude = command.get("amplitude", svar)
                length = command.get("length", svar)
                name = command["name"]
                if invert and command["cycle"]:
                    if name[0] == "-":
                        name = name[1:]
                    else:
                        name = "-" + name
                play(name * amp(amplitude), command["element"], duration=length)

            case "cw":
                amplitude = command.get("amplitude", svar)
                length = command.get("length", svar)
                play("cw" * amp(amplitude), command["element"], duration=length)

            case "wait":
                duration = command.get("length", svar)
                wait(duration)

            case "laser":
                play(command["mode"], command["channel"], duration=command["length"])

            case "measure":
                measure(
                    command["mode"],
                    command["channel"],
                    None,
                    time_tagging.analog(times, command["meas_len"], counts),
                )
                save(counts, counts_st)

            case "align":
                return align()

    def _reference_counts(self, times, counts, counts_st, pi_amp):
        """
        Wrapper for measuring reference counts. Plays qua commands, can only be called from
        within a qua program.

        """

        wait(self.config.wait_between_runs)

        align()

        play("x180" * amp(pi_amp), "NV")  # Pi-pulse toggle
        align()

        if self.measure_delay > 0:
            wait(self.measure_delay, self.measure_channel)
            play("laser_ON", self.laser_channel, duration=self.measure_len)
        else:
            play("laser_ON", self.laser_channel)
        measure(self.measure_mode, self.measure_channel, None, time_tagging.analog(times, self.measure_len, counts))

        save(counts, counts_st)  # save counts

        wait(self.config.wait_between_runs, self.laser_channel)

    def create_experiment(self, n_avg, measure_contrast):
        """
        Creates the Quantum Machine program for the experiment, and returns the
        experiment object as a qua `program`. This is used by the `execute_experiment` and
        `simulate_experiment` methods.

        Args:
            n_avg (int, optional): Number of averages for each data acquisition point.
            measure_contrast (bool): If True, only the |0> state is measured, if False, both |0> and |1> are measured.

        Returns:
            program: The QUA program for the experiment defined by this class's commands.
        """

        with program() as experiment:

            # define the variables and datastreams
            counts0 = declare(int)
            counts0_st = declare_stream()
            counts_ref0 = declare(int)
            counts_ref0_st = declare_stream()

            if not measure_contrast:
                counts1 = declare(int)
                counts1_st = declare_stream()
                counts_ref1 = declare(int)
                counts_ref1_st = declare_stream()

            times = declare(int, size=100)  # QUA vector for storing time-tags

            if self.use_fixed:
                var = declare(fixed)
            else:
                var = declare(int)

            n = declare(int)  # averaging var
            n_st = declare_stream()  # stream for number of iterations

            # start the experiment
            if self.initialize:
                play("laser_ON", self.laser_channel)

                wait(self.config.wait_for_initialization, self.laser_channel)

            with for_(n, 0, n < n_avg, n + 1):  # averaging loop
                with for_(*from_array(var, self.var_vec)):  # scanning loop

                    # do the sequence as defined by the commands, measure |0>
                    for command in self.commands:
                        self._translate_command(command, var, times, counts0, counts0_st, invert=False)

                    # measure reference counts for |0>
                    self._reference_counts(times, counts_ref0, counts_ref0_st, pi_amp=0)

                    # redo above sequennce with a pi-pulse, measuring |1>, if desired
                    if not measure_contrast:
                        for command in self.commands:
                            self._translate_command(command, var, times, counts1, counts1_st, invert=True)

                        self._reference_counts(times, counts_ref1, counts_ref1_st, pi_amp=1)

                    # always end with a wait and saving the number of iterations

                save(n, n_st)

            with stream_processing():
                # save the data from the datastream as 1D arrays on the OPx, with a
                # built in running average
                counts0_st.buffer(len(self.var_vec)).average().save("counts0")
                counts_ref0_st.buffer(len(self.var_vec)).average().save("counts_ref0")
                if not measure_contrast:
                    counts1_st.buffer(len(self.var_vec)).average().save("counts1")
                    counts_ref1_st.buffer(len(self.var_vec)).average().save("counts_ref1")
                n_st.save("iteration")

        return experiment

    def simulate_experiment(self, sim_length=10_000, n_avg=100_000, measure_contrast=True):
        """
        Simulates the experiment using the configured experiment defined by this class based on the current
        config defined by this instance's `config` attribute. The simulation returns the generated waveforms
        of the experiment up to the duration `sim_length` in ns. Useful for checking the timings before running
        on hardware.

        Parameters:
            sim_length (int, optional): The duration of the simulation in ns. Defaults to 10_000.
            n_avg (int, optional): The number of averages per point. Defaults to 100_000.
            measure_contrast (bool): If True, only the |0> state is measured, if False, both |0> and |1> are measured.

        Raises:
            ValueError: Throws an error if insufficient details about the experiment are defined.
        """
        if len(self.commands) == 0:
            raise ValueError("No commands have been added to the experiment.")
        if self.var_vec is None:
            raise ValueError("No inner loop has been defined, invalid sweep.")

        expt = self.create_experiment(n_avg=n_avg, measure_contrast=measure_contrast)
        simulation_config = SimulationConfig(duration=sim_length // 4)
        job = self.config.qmm.simulate(self.config.config, expt, simulation_config)
        job.get_simulated_samples().con1.plot()
        plt.show()
        return job

    def execute_experiment(self, n_avg=100_000, measure_contrast=True, live_plot=True):
        """
        Executes the experiment using the configuration defined by this class. The results are
        stored in the class instance. The results will be visualized live, but this can be
        disabled by setting `live_plot=False`. For each value in the variable `var_vec`, the experiment
        will be run `n_avg` times.

        Parameters:
            n_avg (int, optional): The number of averages per point. Defaults to 100_000.
            measure_contrast (bool): If True, only the |0> state is measured, if False, both |0> and |1> are measured.
            live_plot (bool): If True, the results will be plotted live as they are generated. Defaults to True.

        Raises:
            ValueError: Throws an error if insufficient details about the experiment are defined.
        """
        if len(self.commands) == 0:
            raise ValueError("No commands have been added to the experiment.")
        if self.var_vec is None:
            raise ValueError("No variable vector has been defined.")

        expt = self.create_experiment(n_avg=n_avg, measure_contrast=measure_contrast)

        # Open the quantum machine
        qm = self.config.qmm.open_qm(self.config.config)

        # turn on the microwave control
        self.config.enable_mw1()
        self.config.enable_mw2()

        # Send the QUA program to the OPX, which compiles and executes it
        job = qm.execute(expt)

        # set the data lists being generated to later fetch
        data_list = ["counts0", "counts_ref0"]
        if not measure_contrast:
            data_list.extend(["counts1", "counts_ref1"])
        data_list.append("iteration")

        mode = "live" if live_plot else "wait_for_all"
        # create the fetch tool
        results = fetching_tool(job, data_list=data_list, mode=mode)

        if live_plot:

            fig = plt.figure()
            interrupt_on_close(fig, job)  # Interrupts the job when closing the figure

            while results.is_processing():
                # Fetch results
                if measure_contrast:
                    counts0, counts_ref0, iteration = results.fetch_all()
                else:
                    counts0, counts_ref0, counts1, counts_ref1, iteration = results.fetch_all()
                # Progress bar
                progress_counter(iteration, n_avg, start_time=results.get_start_time())
                # Plot data
                plt.cla()
                plt.plot(
                    self.var_vec * self.x_axis_scale,
                    counts0 / 1000 / (self.measure_len * 1e-9),
                    label="sig0",
                )
                plt.plot(
                    self.var_vec * self.x_axis_scale,
                    counts_ref0 / 1000 / (self.measure_len * 1e-9),
                    label="ref0",
                )

                if not measure_contrast:
                    plt.plot(
                        self.var_vec * self.x_axis_scale,
                        counts1 / 1000 / (self.measure_len * 1e-9),
                        label="sig1",
                    )
                    plt.plot(
                        self.var_vec * self.x_axis_scale,
                        counts_ref1 / 1000 / (self.measure_len * 1e-9),
                        label="ref1",
                    )

                plt.xlabel(self.x_axis_label)
                plt.ylabel(self.y_axis_label)
                plt.title(self.plot_title)
                plt.legend()
                plt.pause(0.1)
        else:
            # Get results from QUA program
            results.wait_for_all_values()
            # Fetch results
            if measure_contrast:
                counts0, counts_ref0, iteration = results.fetch_all()
            else:
                counts0, counts_ref0, counts1, counts_ref1, iteration = results.fetch_all()

        # store the final results
        self.counts0 = counts0
        self.counts_ref0 = counts_ref0
        if not measure_contrast:
            self.counts1 = counts1
            self.counts_ref1 = counts_ref1
        self.iteration = iteration

        # turn off the microwave control, close connection
        self.config.disable_mw1()
        self.config.disable_mw2()
        qm.close()

    def save(self, filename=None):
        """
        Saves the experiment configuration to a JSON file.

        Args:
            filename (string): Path to the JSON file to save, defaults to a timestamped filename if
                none is provided
        """
        filename = (
            self.file_prefix + f"_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json" if filename is None else filename
        )
        try:
            with open(filename, "w") as f:
                attributes = {k: v for k, v in self.__dict__.items() if k != "config"}
                attributes["config"] = self.config.to_dict()
                json.dump(attributes, f, indent=4, cls=NumpyEncoder)
        except (OSError, IOError) as e:
            print(f"Error saving file: {e}")

    @staticmethod
    def load(filename):
        """
        Loads the experiment configuration from a JSON file.

        Args:
            filename (string): Path to the JSON file to load
        """
        expt = Experiment.__new__(Experiment)
        try:
            with open(filename, "r") as f:
                attributes = json.load(f)
            for k, v in attributes.items():
                if k == "config":
                    expt.config = ConfigNV.from_dict(v)
                else:
                    expt.__dict__[k] = v
        except (OSError, IOError, FileNotFoundError) as e:
            print(f"Error loading file: {e}")

        return expt
