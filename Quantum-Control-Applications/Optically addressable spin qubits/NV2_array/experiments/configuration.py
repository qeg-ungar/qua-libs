from pathlib import Path
import numpy as np
from qualang_tools.units import unit
from qualang_tools.plot import interrupt_on_close
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
import plotly.io as pio

from SG384 import SG384Control

pio.renderers.default = "browser"

#######################
# AUXILIARY FUNCTIONS #
#######################
u = unit(coerce_to_integer=True)


# IQ imbalance matrix
def IQ_imbalance(g, phi):
    """
    Creates the correction matrix for the mixer imbalance caused by the gain and phase imbalances, more information can
    be seen here:
    https://docs.qualang.io/libs/examples/mixer-calibration/#non-ideal-mixer
    :param g: relative gain imbalance between the 'I' & 'Q' ports. (unit-less), set to 0 for no gain imbalance.
    :param phi: relative phase imbalance between the 'I' & 'Q' ports (radians), set to 0 for no phase imbalance.
    """
    c = np.cos(phi)
    s = np.sin(phi)
    N = 1 / ((1 - g**2) * (2 * c**2 - 1))
    return [float(N * x) for x in [(1 - g) * c, (1 + g) * s, (1 - g) * s, (1 + g) * c]]


######################
# Network parameters #
######################
qop_ip = "18.25.10.244"  # Write the OPX IP address
cluster_name = "QM_NV2"  # Write your cluster_name if version >= QOP220
qop_port = None  # Write the QOP port if version < QOP220

#############
# Save Path #
#############
# Path to save data
# AU: changed default path to be one level up in Data folder
save_dir = Path(__file__).parent.resolve().parent / "Data"
save_dir.mkdir(parents=True, exist_ok=True)

# Build additional files dict, only including files that exist
default_additional_files = {
    str(Path(__file__).resolve()): Path(__file__).name,
}
# Add optional files if they exist
weights_path = Path(__file__).parent / "optimal_weights.npz"
if weights_path.exists():
    default_additional_files[str(weights_path)] = "optimal_weights.npz"

############################
# Set SG384 configuration #
############################
NV_LO_freq = 1.794 * u.GHz  # aligned 355 G #1.795
NV_LO_amp = -16  # in dBm
sg384 = SG384Control("TCPIP0::18.25.11.6::5025::SOCKET")
sg384.set_amplitude(NV_LO_amp)
sg384.set_frequency(NV_LO_freq)
sg384.ntype_on(1)
sg384.set_modulation_state("ON")
sg384.set_modulation_type("IQ")

#####################
# OPX configuration #
#####################
# Set octave_config to None if no octave is present
octave_config = None

sampling_rate = int(1e9)  # needed in some scripts

# Frequencies
NV_IF_freq = 80.06 * u.MHz  # NV IF frequency

# Pulses lengths
initialization_len_1 = 3000 * u.ns  # NV ensemble calibrated with  2026-02-10\#119_calibrate_delays_185952
meas_len_1 = 384 * u.ns  # 500 #calibrated at 0.48 mW with 2026-02-14\#218_calibrate_readout_163113
long_meas_len_1 = 5_000 * u.ns

initialization_len_2 = 3000 * u.ns 
long_meas_len_2 = 5_000 * u.ns
meas_len_2 = 384 * u.ns  # 500 #calibrated at 0.48 mW with 2026-02-14\#218_calibrate_readout_163113
readout_len_SPAD = 384 * u.ns

# Relaxation time from the metastable state to the ground state after during initialization
relaxation_time = 300 * u.ns
wait_for_initialization = 5 * relaxation_time

# MW parameters
mw_amp_NV = 0.052  # in units of volts
mw_len_NV = 500 * u.ns

x180_amp_NV = 0.177  # in units of volts #calibrate with #2026-02-10\#108_power_rabi_175847
x180_len_NV = 148 * u.ns  # in units of ns #calibrate with #2026-02-10\#105_time_rabi_174509

# x180_amp_NV = 0.5*.06275  # in units of volts
# x180_len_NV = 1000 * u.ns  # in units of ns


x90_amp_NV = x180_amp_NV / 2  # in units of volts
x90_len_NV = x180_len_NV  # in units of ns

# RF parameters
rf_frequency = 10 * u.MHz
rf_amp = 0.1
rf_length = 1000

# Readout parameters
signal_threshold_1 = -8_00  # ADC units, to convert to volts divide by 4096 (12 bit ADC)
signal_threshold_2 = -8_00  #2_000  #ADC units, to convert to volts divide by 4096 (12 bit ADC)

# Delays #calibrated with 2026-02-10\#115_calibrate_delays_184951
#did not add extra delay relative to start of laser, according to one-note NV ensemble calibration 20260210
#optimal start start time is 519 ns and duration of 384 ns (average of 372 and 396 from 2 methods)
#detection_delay_1 = 324 * u.ns  #running 'calibrate_delays' shows laser start at 500 ns
detection_delay_1 = 344 * u.ns #added 20 ns delay on 2026-02-26 to account for delay in laser rise time, so that now detection starts at 500- 20 ns in calibration script
#detection_delay_2 = 5628 * u.ns
detection_delay_2 = 1440 * u.ns #running 'calibrate_delays' shows laser start at 500 ns #see on-note 'SNR and delays with high power laser'
detection_delay_2 = 1540 * u.ns #20260226 added 100 ns delay, so rise of laser starts at 400 ns in calibration script

laser_delay_1 = 196 * u.ns
laser_delay_2 = 0 * u.ns

mw_delay_1 = 0 * u.ns
mw_delay_2 = 1000 * u.ns
mw_delay = mw_delay_2

SPAD_delay = 17800 * u.ns #20.8 us min repetition time - 3 us laser pulse

rf_delay = 0 * u.ns

wait_between_runs = 500 * u.ns  # calibrated 2026-02-10 with CW-ODMR ref

config = {
    "controllers": {
        "con1": {
            "analog_outputs": {
                1: {"offset": -0.02, "delay": mw_delay},  # NV I
                2: {"offset": -0.02, "delay": mw_delay},  # NV Q
                3: {"offset": 0.0, "delay": rf_delay},  # RF
            },
            "digital_outputs": {
                1: {},  # AOM/Laser
                2: {},  # AOM/Laser
                3: {},  # SPCM1 - indicator
                4: {},  # SPCM2 - indicator
                5: {},  # SPAD array 
            },
            "analog_inputs": {
                1: {"offset": 0},  # SPCM1
                2: {"offset": 0},  # SPCM2
            },
        }
    },
    "elements": {
        "NV": {
            "mixInputs": {
                "I": ("con1", 1),
                "Q": ("con1", 2),
                "lo_frequency": NV_LO_freq,
                "mixer": "mixer_NV",
            },
            "intermediate_frequency": NV_IF_freq,
            "operations": {
                "cw": "const_pulse",
                "x180": "x180_pulse",
                "x90": "x90_pulse",
                "-x90": "-x90_pulse",
                "-y90": "-y90_pulse",
                "y90": "y90_pulse",
                "y180": "y180_pulse",
            },
        },
        "RF": {
            "singleInput": {"port": ("con1", 3)},
            "intermediate_frequency": rf_frequency,
            "operations": {
                "const": "const_pulse_single",
            },
        },
        "AOM1": {
            "digitalInputs": {
                "marker": {
                    "port": ("con1", 1),
                    "delay": laser_delay_1,
                    "buffer": 0,
                },
            },
            "operations": {
                "laser_ON": "laser_ON_1",
            },
        },
        "AOM2": {
            "digitalInputs": {
                "marker": {
                    "port": ("con1", 2),
                    "delay": laser_delay_2,
                    "buffer": 0,
                },
            },
            "operations": {
                "laser_ON": "laser_ON_2",
            },
        },
        "SPAD": {
            "digitalInputs": {
                "marker": {
                    "port": ("con1", 5),
                    "delay": detection_delay_2,
                    "buffer": 0,
                },
            },
            "operations": {
                "readout_SPAD": "readout_pulse_SPAD",
            },
        },
        "SPCM1": {
            "singleInput": {"port": ("con1", 1)},  # not used
            "digitalInputs": {  # for visualization in simulation
                "marker": {
                    "port": ("con1", 3),
                    "delay": detection_delay_1,
                    "buffer": 0,
                },
            },
            "operations": {
                "readout": "readout_pulse_1",
                "long_readout": "long_readout_pulse_1",
            },
            "outputs": {"out1": ("con1", 1)},
            "timeTaggingParameters": {
                "signalThreshold": signal_threshold_1,  # ADC units
                "signalPolarity": "Below",
                "derivativeThreshold": -2_000,
                "derivativePolarity": "Above",
            },
            "time_of_flight": detection_delay_1,
            "smearing": 0,
        },
        "SPCM2": {
            "singleInput": {"port": ("con1", 1)},  # not used
            "digitalInputs": {  # for visualization in simulation
                "marker": {
                    "port": ("con1", 4),
                    "delay": detection_delay_2,
                    "buffer": 0,
                },
            },
            "operations": {
                "readout": "readout_pulse_2",
                "long_readout": "long_readout_pulse_2",
            },
            "outputs": {"out1": ("con1", 1)},
            "timeTaggingParameters": {
                "signalThreshold": signal_threshold_2,  # ADC units
                "signalPolarity": "Below",
                "derivativeThreshold": -2_000,
                "derivativePolarity": "Above",
            },
            "time_of_flight": detection_delay_2,
            "smearing": 0,
        },
    },
    "pulses": {
        "const_pulse": {
            "operation": "control",
            "length": mw_len_NV,
            "waveforms": {"I": "cw_wf", "Q": "zero_wf"},
        },
        "x180_pulse": {
            "operation": "control",
            "length": x180_len_NV,
            "waveforms": {"I": "x180_wf", "Q": "zero_wf"},
        },
        "x90_pulse": {
            "operation": "control",
            "length": x90_len_NV,
            "waveforms": {"I": "x90_wf", "Q": "zero_wf"},
        },
        "-x90_pulse": {
            "operation": "control",
            "length": x90_len_NV,
            "waveforms": {"I": "minus_x90_wf", "Q": "zero_wf"},
        },
        "-y90_pulse": {
            "operation": "control",
            "length": x90_len_NV,
            "waveforms": {"I": "zero_wf", "Q": "minus_x90_wf"},
        },
        "y90_pulse": {
            "operation": "control",
            "length": x90_len_NV,
            "waveforms": {"I": "zero_wf", "Q": "x90_wf"},
        },
        "y180_pulse": {
            "operation": "control",
            "length": x180_len_NV,
            "waveforms": {"I": "zero_wf", "Q": "x180_wf"},
        },
        "const_pulse_single": {
            "operation": "control",
            "length": rf_length,  # in ns
            "waveforms": {"single": "rf_const_wf"},
        },
        "laser_ON_1": {
            "operation": "control",
            "length": initialization_len_1,
            "digital_marker": "ON",
        },
        "laser_ON_2": {
            "operation": "control",
            "length": initialization_len_2,
            "digital_marker": "ON",
        },
        "readout_pulse_1": {
            "operation": "measurement",
            "length": meas_len_1,
            "digital_marker": "ON",
            "waveforms": {"single": "zero_wf"},
        },
        "long_readout_pulse_1": {
            "operation": "measurement",
            "length": long_meas_len_1,
            "digital_marker": "ON",
            "waveforms": {"single": "zero_wf"},
        },
        "readout_pulse_2": {
            "operation": "measurement",
            "length": meas_len_2,
            "digital_marker": "ON",
            "waveforms": {"single": "zero_wf"},
        },
        "long_readout_pulse_2": {
            "operation": "measurement",
            "length": long_meas_len_2,
            "digital_marker": "ON",
            "waveforms": {"single": "zero_wf"},
        },
        "readout_pulse_SPAD": {
            "operation": "control",
            "length": readout_len_SPAD,
            "digital_marker": "ON",
        },
    },
    "waveforms": {
        "cw_wf": {"type": "constant", "sample": mw_amp_NV},
        "rf_const_wf": {"type": "constant", "sample": rf_amp},
        "x180_wf": {"type": "constant", "sample": x180_amp_NV},
        "x90_wf": {"type": "constant", "sample": x90_amp_NV},
        "minus_x90_wf": {"type": "constant", "sample": -x90_amp_NV},
        "zero_wf": {"type": "constant", "sample": 0.0},
    },
    "digital_waveforms": {
        "ON": {"samples": [(1, 0)]},  # [(on/off, ns)]
        "OFF": {"samples": [(0, 0)]},  # [(on/off, ns)]
    },
    "mixers": {
        "mixer_NV": [
            {"intermediate_frequency": NV_IF_freq, "lo_frequency": NV_LO_freq, "correction": IQ_imbalance(0.03, -0.05)},
        ],
    },
}
