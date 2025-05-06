"""
        POWER RABI
The program consists in playing a mw pulse (3*pi_len) and measure the photon counts received by the SPCM
across varying mw pulse amplitudes.
The sequence is repeated without playing the mw pulses to measure the dark counts on the SPCM.

The data is then post-processed to determine the pi pulse amplitude for the specified duration.

Prerequisites:
    - Ensure calibration of the different delays in the system (calibrate_delays).
    - Having updated the different delays in the configuration.
    - Having updated the NV frequency, labeled as "NV_IF_freq", in the configuration.
    - Set the desired pi pulse duration, labeled as "mw_len_NV", in the configuration

Next steps before going to the next node:
    - Update the pi pulse amplitude, labeled as "mw_amp_NV", in the configuration.
"""

from qm import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from configuration_NV2QEG import *
from qualang_tools.loops import from_array
from scipy.optimize import curve_fit
from datetime import datetime
import os

data_path = "C:\\Users\\SPUD1\\Documents\\qm_workspace\\saved_experiments\\"
script_name = "power_Rabi"
var_name = "pi_amp_NV"
###################
# The QUA program #
###################

#a_vec = np.arange(0, 2.0, 0.02)  # The amplitude pre-factor vector
a_vec = np.arange(0, 2.0, 0.03)  # pnly works with old version 1.1.6 

n_avg = 500_000  # number of iterations
# n_avg = 1_000_000  # number of iterations

# pi_len_scale = 1
pi_len_scale = 3  # to calibrate 3 pi-pulse for better sensitivity to over-rotation

boolSave = True

with program() as power_rabi:
    counts = declare(int)  # variable for number of counts
    times = declare(int, size=100)  # QUA vector for storing the time-tags
    a = declare(fixed)  # variable to sweep over the amplitude
    n = declare(int)  # variable to for_loop
    counts_st = declare_stream()  # stream for counts
    counts_dark_st = declare_stream()  # stream for counts
    n_st = declare_stream()  # stream to save iterations

    # Spin initialization
    play("laser_ON", "AOM1")
    wait(wait_for_initialization * u.ns, "AOM1")

    # Power Rabi sweep
    with for_(n, 0, n < n_avg, n + 1):
        with for_(*from_array(a, a_vec)):
            # Play the Rabi pulse with varying amplitude
            play(
                "x180" * amp(a), "NV", duration=pi_len_scale * x180_len_NV * u.ns
            )  # rotate by 540 deg to magnify any error 'a' is a pre-factor to the amplitude defined in the config ("mw_amp_NV")
            align()  # Play the laser pulse after the mw pulse
            play("laser_ON", "AOM1")
            # Measure and detect the photons on SPCM1
            measure("readout", "SPCM1", None, time_tagging.analog(times, meas_len_1, counts))
            save(counts, counts_st)  # save counts

            # Wait and align all elements before measuring the dark events
            wait(wait_between_runs * u.ns)
            align()

            # Play the Rabi pulse with zero amplitude
            play("x180" * amp(0), "NV", duration=pi_len_scale * x180_len_NV * u.ns)
            align()  # Play the laser pulse after the mw pulse
            play("laser_ON", "AOM1")
            # Measure and detect the dark counts on SPCM1
            measure("readout", "SPCM1", None, time_tagging.analog(times, meas_len_1, counts))
            save(counts, counts_dark_st)  # save dark counts
            wait(wait_between_runs * u.ns)  # wait in between iterations

        save(n, n_st)  # save number of iteration inside for_loop

    with stream_processing():
        # Cast the data into a 1D vector, average the 1D vectors together and store the results on the OPX processor
        counts_st.buffer(len(a_vec)).average().save("counts")
        counts_dark_st.buffer(len(a_vec)).average().save("counts_dark")
        n_st.save("iteration")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qop_ip, cluster_name=cluster_name)

#######################
# Simulate or execute #
#######################
simulate = False

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, power_rabi, simulation_config)
    job.get_simulated_samples().con1.plot()
    plt.show()
else:
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Open communication with SRS and set LO parameters
    #sg384_NV.set_amplitude(NV_LO_amp)
    #sg384_NV.set_frequency(NV_LO_freq)
    #sg384_NV.ntype_on(1)
    #sg384_NV.do_set_Modulation_State("ON")
    #sg384_NV.do_set_modulation_type("IQ")
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(power_rabi)
    # Get results from QUA program
    results = fetching_tool(job, data_list=["counts", "counts_dark", "iteration"], mode="live")
    # Live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)  # Interrupts the job when closing the figure

    while results.is_processing():
        # Fetch results
        counts, counts_dark, iteration = results.fetch_all()
        # Progress bar
        progress_counter(iteration, n_avg, start_time=results.get_start_time())
        # Plot data
        plt.cla()
        plt.plot(a_vec * x180_amp_NV, counts / 1000 / (meas_len_1 * 1e-9), label="photon counts")
        plt.plot(a_vec * x180_amp_NV, counts_dark / 1000 / (meas_len_1 * 1e-9), label="dark_counts")
        plt.xlabel("Rabi pulse amplitude [V]")
        plt.ylabel("Intensity [kcps]")
        plt.title("Power Rabi")
        plt.legend()
        plt.pause(0.1)
    # Turn off SRS
    #sg384_NV.ntype_on(0)

# data anlayis to extract pi_amp from cosine-decay fit
# y = (counts - counts_dark) / counts_dark
# x = a_vec * x180_amp_NV

# plt.figure()
# plt.plot(x, y, "b-", label="data")


# def f_cos_decay(x, a1, a2, a3, a4, a5, a6):
#     return a1 * np.cos(2 * np.pi * x / a2 + 2 * np.pi * a3) * np.exp(-a4 * x) + a5 * x + a6


# a1_0 = 0.5 * np.abs(np.max(y) - np.min(y))  # amplitude
# if pi_len_scale > 1:
#     a2_0 = (2 / 3) * pi_amp_NV
# else:
#     a2_0 = pi_amp_NV
# a3_0 = 0
# a4_0 = 0
# a5_0 = -0.25 * abs(np.mean(y)) / np.mean(x)
# # a5_0 = 0
# a6_0 = np.mean(y)

# p0 = a1_0, a2_0, a3_0, a4_0, a5_0, a6_0
# popt, pcov = curve_fit(f_cos_decay, x, y, p0)
# yfit = f_cos_decay(x, *popt)
# print("Guess fit parameters for a1*np.cos(2*np.pi*x/a2 + 2*np.pi*a3)*np.exp(-a4*x) + a5*x + a6:\n", p0)
# print("Calculated fit parameters:\n", popt)
# print("Standard deviation of fit parameters:")
# print(np.sqrt(np.diag(pcov)))
# # if pi_len_scale > 1:
# #    print('pi_amp from fit:');print((3/2)*popt[1])
# # else:
# #    print('pi_amp from fit:');print(popt[1])

# xfine = np.linspace(np.min(x), np.max(x), 1000)
# plt.plot(x, y, "b-")
# plt.plot(xfine, f_cos_decay(xfine, *popt), "r-", label="sloped cosine decay fit")
# plt.title("Power Rabi")
# plt.xlabel("Rabi pulse amplitude [V]")
# plt.ylabel("contrast")
# plt.legend(loc="upper right")
# plt.show(block=False)

# xfine_loc = xfine[np.where((xfine > 0.9 * pi_amp_NV) & (xfine < 1.1 * pi_amp_NV))[0]]
# yfit_loc = f_cos_decay(xfine_loc, *popt)
# x_min = xfine_loc[np.argmin(yfit_loc)]
# print("pi_amp from fit:")
# print(x_min)

# # generate data array and time stamp for saving
# data = [a_vec * pi_amp_NV, counts / (meas_len_1 * 1e-9), counts_dark / (meas_len_1 * 1e-9)]

# num_ave = str(iteration + 1)
# now = datetime.now()  # current date and time
# timestamp = now.strftime("%m_%d_%Y_%H_%M_%S")

# if boolSave:
#     np.savetxt(data_path + script_name + "_" + var_name + "_" + num_ave + "_" + timestamp + ".txt", data)
