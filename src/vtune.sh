source /opt/intel/vtune_amplifier_xe/amplxe-vars.sh
source /opt/intel/vtune_amplifier_xe/sep_vars.sh

amplxe-cl -collect hotspots -- ./train.sh

# amplxe-cl -collect memory-access -- ./train.sh

# amplxe-cl -collect-with runsa -knob event-config=? -- ./train.sh
# amplxe-gui
