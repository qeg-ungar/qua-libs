"""
        CW Optically Detected Magnetic Resonance (ODMR)
The program consists in playing a mw pulse and the readout laser pulse simultaneously to extract
the photon counts received by the SPCM across varying intermediate frequencies.

The data is then post-processed to determine the spin resonance frequency.
This frequency can be used to update the NV intermediate frequency in the configuration under "NV_IF_freq".

Prerequisites:
    - Ensure calibration of the different delays in the system (calibrate_delays).
    - Update the different delays in the configuration

Next steps before going to the next node:
    - Update the NV frequency, labeled as "NV_IF_freq", in the configuration.
"""

from qm import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
import matplotlib.pyplot as plt
from configuration import *
from qualang_tools.results.data_handler import DataHandler
import rpyc
import threading
import time

QUDI_NAMESPACE_PORT = 18861
QUDI_SWITCH_MODULE = "qm_switch_ni"
QUDI_SWITCH_NAME = "One"
QUDI_TRIGGER_STATE = "High"

##################
#   Parameters   #
##################
# Parameters Definition
#t_vec = np.arange(4, 1500, 30)  # Pulse durations in clock cycles (4ns) #0.5 MHz Rabi
t_vec = np.arange(4, 500, 15)  # Pulse durations in clock cycles (4ns) #1.67 MHz Rabi

mw_switch_buffer = 250 #in clock cycles (1 us)
mw_switch_len_idle = np.max(t_vec)/2 + mw_switch_buffer #in clock cycles

#n_avg = 10_000_000  # number of averages
n_avg = 600_000  # number of averages
#n_avg = 500_000  # number of averages

# Determine reference readout during single laser pulse
reference_wait = initialization_len_1 // 4 - 2 * meas_len_1 // 4 - 25  # in clock cycles
reference_readout = reference_wait >= 4

# Data to save
save_data_dict = {
    "n_avg": n_avg,
    "t_vec": t_vec,
    "config": config,
}

###################
# The QUA program #
###################

wait_trigger = True

def get_prog(wait_trigger=wait_trigger):
    with program() as pulsed_odmr:
        times = declare(int, size=100)  # QUA vector for storing the time-tags
        counts = declare(int)  # variable for number of counts
        counts_st = declare_stream()  # stream for counts
        counts_ref_st = declare_stream()  # stream for counts
        t = declare(int)  # variable to sweep over the pulse duration
        n = declare(int)  # number of iterations
        n_st = declare_stream()  # stream for number of iterations
        if wait_trigger:
            trigger_flag = declare(bool, value=False)
            io1_val = declare(fixed)
            with while_(~trigger_flag): #run sequence while waiting for trigger to keep setup stable
                play("mw_switch_ON", "MW_switch", duration = mw_switch_len_idle)
                play("x180" * amp(1), "NV")
                align()
                wait(wait_after_gradient * u.ns)
                play("laser_ON", "AOM1") #dont trigger AOM2
                #play("readout_SPAD", "SPAD")
                align()
                wait(2 * SPAD_delay * u.ns) #to keep duty cycle approx the same as ODMR
                #wait(mw_switch_len) #to keep duty cycle approx the same as ODMR
                play("x180" * amp(0), "NV")
                align()
                play("laser_ON", "AOM1")
                #play("readout_SPAD", "SPAD")
                align()
                wait(SPAD_delay * u.ns)
                assign(io1_val, IO1)
                assign(trigger_flag, io1_val > 0)
        align()

        with for_(n, 0, n < n_avg, n + 1):
            with for_(*from_array(t, t_vec)):
                # Update the duration of the mw pulse
                #turn on MW switch
                play("mw_switch_ON", "MW_switch", duration = t + mw_switch_buffer)
                # Play the mw pulse...
                play("x180" * amp(1), "NV", duration=t)
                align()
                wait(wait_after_gradient * u.ns)
                # ... and the laser pulse simultaneously (the laser pulse is delayed by 'laser_delay_1')
                play("laser_ON", "AOM2")
                #measure("readout", "SPCM2", time_tagging.analog(times, meas_len_2, counts))
                #wait(init_delay * u.ns, "SPAD")  # so readout don't catch the first part of spin reinitialization
                # Measure and detect the photons on SPCM1
                play("readout_SPAD", "SPAD")
                #measure("long_readout", "SPCM2", time_tagging.analog(times, readout_len, counts))

                #save(counts, counts_st)  # save counts on stream
                align()
                wait(2 * SPAD_delay * u.ns) #to keep duty cycle approx the same as ODMR
                #wait(mw_switch_len) #to keep duty cycle approx the same as ODMR

                # Play the mw pulse with zero amplitude...
                play("x180" * amp(0), "NV", duration=t)
                align()
                # ... and the laser pulse simultaneously (the laser pulse is delayed by 'laser_delay_1')
                play("laser_ON", "AOM2")
                #measure("readout", "SPCM2", time_tagging.analog(times, meas_len_2, counts))
                #wait(init_delay * u.ns, "SPAD")  # so readout don't catch the first part of spin reinitialization
                # Measure and detect the photons on SPCM1
                play("readout_SPAD", "SPAD")
                #measure("long_readout", "SPCM1", time_tagging.analog(times, readout_len, counts))

                #save(counts, counts_ref_st)  # save counts on stream
                align()

                wait(SPAD_delay * u.ns)

                save(n, n_st)  # save number of iteration inside for_loop

        with stream_processing():
            # Cast the data into a 1D vector, average the 1D vectors together and store the results on the OPX processor
            #counts_st.buffer(len(f_vec)).average().save("counts")
            #counts_ref_st.buffer(len(f_vec)).average().save("counts_ref")
            n_st.save("iteration")
        return pulsed_odmr

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qop_ip, cluster_name=cluster_name, octave=octave_config)

#######################
# Simulate or execute #
#######################
simulate = False

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    # Simulate blocks python until the simulation is done
    job = qmm.simulate(config, get_prog(), simulation_config)
    # Get the simulated samples
    samples = job.get_simulated_samples()
    # Plot the simulated samples
    samples.con1.plot()
    # Get the waveform report object
    waveform_report = job.get_simulated_waveform_report()
    # Cast the waveform report to a python dictionary
    waveform_dict = waveform_report.to_dict()
    # Visualize and save the waveform report
    waveform_report.create_plot(samples, plot=True, save_path=str(Path(__file__).resolve()))
else:
    def watch_qudi_trigger(qm, stop_event):
        try:
            protocol_config = {
                'allow_all_attrs': True,
                'allow_setattr': True,
                'allow_delattr': True,
                'allow_pickle': True,
                'sync_request_timeout': 3600,
            }
            conn = rpyc.connect("localhost", QUDI_NAMESPACE_PORT, config=protocol_config)
            try:
                namespace = conn.root.get_namespace_dict()
                switch = namespace[QUDI_SWITCH_MODULE]
                print(f"[watch_qudi_trigger] connected, polling '{QUDI_SWITCH_MODULE}'...")
                while not stop_event.is_set():
                    if switch.get_state(QUDI_SWITCH_NAME) == QUDI_TRIGGER_STATE:
                        print("[watch_qudi_trigger] trigger detected, setting IO1=True")
                        qm.set_io1_value(True)
                        break
                    time.sleep(0.01)
            finally:
                conn.close()
        except Exception as e:
            print(f"[watch_qudi_trigger] ERROR: {e}")

    # Open the quantum machine
    qm = qmm.open_qm(config, close_other_machines=True)

    script_name = Path(__file__).name
    script_path = Path(__file__).resolve()
    data_handler = DataHandler(root_data_folder=save_dir)

    # Create figure once; closing it stops the loop
    fig = plt.figure()
    current_job = [None]
    keep_running = [True]

    def on_close(_):
        keep_running[0] = False
        if current_job[0] is not None:
            try:
                current_job[0].halt()
            except Exception:
                pass

    fig.canvas.mpl_connect('close_event', on_close)

    run_count = 0
    while keep_running[0]:
        print(f"\n--- Starting run {run_count + 1} (waiting for trigger) ---")
        qm.set_io1_value(False)  # reset from previous run

        stop_event = threading.Event()
        if wait_trigger:
            t = threading.Thread(
                target=watch_qudi_trigger,
                args=(qm, stop_event),
                daemon=True,
            )
            t.start()

        job = qm.execute(get_prog())
        current_job[0] = job
        results = fetching_tool(job, data_list=["iteration"], mode="live")
        #results = fetching_tool(job, data_list=["counts", "counts_ref", "iteration"], mode="live")

        while results.is_processing() and keep_running[0]:
            # Fetch results
            (iteration,) = results.fetch_all()
            # Progress bar
            progress_counter(iteration, n_avg, start_time=results.get_start_time())
            # Plot data
            plt.cla()
            #plt.plot((NV_LO_freq * 1 + f_vec) / u.MHz, counts / 1000 / (readout_len * 1e-9), label="signal")
            #plt.plot((NV_LO_freq * 1 + f_vec) / u.MHz, counts_ref / 1000 / (readout_len * 1e-9), label="reference")
            #plt.xlabel("MW frequency [MHz]")
            #plt.ylabel("Intensity [kcps]")
            #plt.title("CW ODMR")
            #plt.legend()
            plt.pause(0.1)

        stop_event.set()

        if not keep_running[0]:
            break

        run_count += 1
        print(f"--- Run {run_count} complete, saving data ---")
        # Turn off SRS output
        #sg384.ntype_on(0)
        # Save results
        save_data_dict.update({"iteration": int(iteration)})
        #save_data_dict.update({"counts_data": counts})
        #save_data_dict.update({"counts_ref_data": counts_ref})
        #save_data_dict.update({"fig_live": fig})
        data_handler.additional_files = {str(script_path): script_name, **default_additional_files}
        data_handler.save_data(data=save_data_dict, name="_".join(script_name.split("_")[1:]).split(".")[0])
plt.show()
# Turn off SRS output
sg384.ntype_on(0)