# Configuration file for NV2 confocal experiments using sprout lighthouse laser and SPCM detection
# initialization and detection parameters are calibrated for NV ensemble array ["16","60"] using 0.48 mW laser power
# channel for laser is AOM2, channel for SPCM detection is SPCM2

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
#NV_LO_freq = 1.808 * u.GHz  # aligned [111] 351 G 
#NV_LO_freq = 1.820 * u.GHz  # aligned [111] 351 G #gradient on with DC bias on MW channel
#NV_LO_amp = -16  # in dBm

NV_LO_freq = 1.821 * u.GHz  #aligned [-1-11] 352 G, x = 12.4 mm, gradient off 20260831
#NV_LO_freq = 1.801 * u.GHz  #aligned [-1-11] 352 G, gradient on 202607
#NV_LO_amp = -4  # in dBm #after adding power splitter for gradient
#NV_LO_amp = -8  # in dBm #with DC bias tee for gradient drive
#NV_LO_amp = -2  #
#NV_LO_amp = -16  # -19 #in dBm low power CW ODMR 
NV_LO_amp = -5  # - #in dBm high power CW ODMR 

#NV_LO_freq = 2.356 * u.GHz  #aligned [-1-11] 152 G
#NV_LO_freq = 2.740 * u.GHz  #other orientations 152 G

#NV_LO_freq = 2.350 * u.GHz  #aligned [111] 152 G
#NV_LO_freq = 2.740 * u.GHz  #other orientations 152 G

#NV_LO_freq = 2.500 * u.GHz  #[111] at x = 19 mm
#NV_LO_freq = 2.435 * u.GHz  #
#NV_LO_freq = 2.85 * u.GHz 

#NV_LO_freq = 2.85 * u.GHz  
#NV_LO_freq = 2.45 * u.GHz  
#NV_LO_freq = 2.5 * u.GHz  


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
#NV_IF_freq = 81.07 * u.MHz  # NV IF frequency (right column gradient on) 
#NV_IF_freq = 79.72 * u.MHz  # NV IF frequency (middle column gradient on)
#NV_IF_freq = 78.66 * u.MHz  # NV IF frequency (left column gradient on)
#NV_IF_freq = 80.50 * u.MHz  # NV IF frequency (gradient off)
NV_IF_freq = 80 * u.MHz  # NV IF frequency


# Pulses lengths
initialization_len_1 = 3000 * u.ns  # NV ensemble calibrated with  2026-02-10\#119_calibrate_delays_185952
meas_len_1 = 384 * u.ns  # 500 #calibrated at 0.48 mW with 2026-02-14\#218_calibrate_readout_163113
long_meas_len_1 = 10_000 * u.ns #5_000 * u.ns

initialization_len_2 = 3000 * u.ns #
#initialization_len_2 = 5000 * u.ns #20260415 NV ensemble at 100 uW see one-note 'SPAD pulsed experiments'
meas_len_2 = 484 * u.ns  # 500 #calibrated at 0.48 mW with 2026-03-03\#4_calibrate_readout_183901
#meas_len_2 = 2600 * u.ns  # 20260415 NV ensemble see one-note 'SPAD pulsed experiments' and 2026-04-15\#60_calibrate_readout_101504
long_meas_len_2 = 10_000 * u.ns

# Relaxation time from the metastable state to the ground state after during initialization
relaxation_time = 300 * u.ns
wait_for_initialization = 5 * relaxation_time

# MW parameters
mw_amp_NV = 0.055  # in units of volts #calibrated with 2026-04-27\#317_power_rabi_141608
#mw_amp_NV = 2*0.055  # in units of volts #calibrated with 2026-04-27\#317_power_rabi_141608
mw_len_NV = 500 * u.ns

#x180_amp_NV = 0.1365  # in units of volts #calibrate with 2026-06-26\#628_power_rabi_150630 #with DC bias T
x180_amp_NV = 0.118  # in units of volts #calibrate with 2026-07-07\#667_power_rabi_124024 #with power splitter
x180_len_NV = 148 * u.ns  # in units of ns 

#x180_amp_NV = .1166*148/500  # in units of volts #calibrated with  2026-06-10\#590_power_rabi_165917
#x180_len_NV = 500 * u.ns  # in units of ns

#x180_amp_NV = .0174  # in units of volts #calibrate with 2026-07-02\#638_power_rabi_140044
#x180_amp_NV = .0157 * (0.5/0.53)  # in units of volts #calibrate with \2026-07-06\#656_power_rabi_gradient_162217 gradient on #20260722 modified based on Rabi SPAD analysis
#x180_len_NV = 1000 * u.ns  # in units of ns

x90_amp_NV = x180_amp_NV / 2  # in units of volts
x90_len_NV = x180_len_NV  # in units of ns

# RF parameters
rf_frequency = 10 * u.MHz
rf_amp = 0.1
rf_length = 1000

# MW switch parameters
mw_switch_len = 2000 * u.ns

# Readout parameters
signal_threshold_1 = -8_00  # ADC units, to convert to volts divide by 4096 (12 bit ADC)
signal_threshold_2 = -8_00  # 2_000  #ADC units, to convert to volts divide by 4096 (12 bit ADC)

# detection_delay_1 = 324 * u.ns  #running '04a_calibrate_delays.py' shows laser start at 500 ns
detection_delay_1 = (
    344 * u.ns
)  # 2026-02-26: added 20 ns delay to account for laser rise time, so that now detection starts at 500- 20 ns in calibration script

# delays for laser 2 calibrate in one-note 'SNR and delays with high power laser'
# detection_delay_2 = 1440 * u.ns #running 'calibrate_delays' shows laser start at 500 ns
detection_delay_2 = (
    1440 + 32
) * u.ns  # 2026-03-03: running 'calibrate_readout' added 32 ns delay, so rise of laser starts at 468 ns in calibration script

laser_delay_1 = 196 * u.ns
laser_delay_2 = 0 * u.ns

mw_delay = 1000 * u.ns

rf_delay = 0 * u.ns

#mw_switch_delay = 240 * u.ns #see one-note 'SPAD/Testing/design RF switch for gradient'
#mw_switch_delay = 500 * u.ns #see one-note 'SPAD/Testing/design RF switch for gradient'
#mw_switch_delay = 1200 * u.ns #see one-note 'SPAD/Testing/design RF switch for gradient'
mw_switch_delay = 0 * u.ns #see one-note 'Pulsed gradient testing'

wait_between_runs = 500 * u.ns  # calibrated 2026-02-10 with CW-ODMR ref

#wait_between_runs_SPAD = (20_800 - 3_000) * u.ns  # testing gradient switch
#wait_between_runs_SPAD = (5_000) * u.ns  # testing gradient switch
#wait_between_runs_SPAD = (500) * u.ns  # testing gradient switch
SPAD_HIT = 20_800 * u.ns  # hardware integration time
wait_between_runs_SPAD = SPAD_HIT - initialization_len_2  # 20.8 us min repetition time - 3 us laser pulse

wait_after_gradient = 3000 * u.ns   #20260722 changed from 2 us to 3 us based on scope

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
        "MW_switch": {
            "digitalInputs": {
                "marker": {
                    "port": ("con1", 1), #not using SPCM1
                    "delay": mw_switch_delay,
                    "buffer": 0,
                },
            },
            "operations": {
                "mw_switch_ON": "mw_switch_ON_1",
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
        "mw_switch_ON_1": {
            "operation": "control",
            "length": mw_switch_len,
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
            {"intermediate_frequency": NV_IF_freq, "lo_frequency": NV_LO_freq, "correction": IQ_imbalance(0.03, 0.05)},
        ],
    },
}
