@echo off
set folderpath=C:\Users\jonora\Documents\cgp_neuron_masters\logfiles\config_complex_4\5\log_config_complex_4_1646334773.3087056
set filepaths=C:\Users\jonora\Documents\cgp_neuron_masters\logfiles\config_complex_4\5\log_config_complex_4_1646334773.3087056\statistics.yml

python C:\Users\jonora\Documents\cgp_neuron_masters\CGP_Neuron_Masters\yaml_file_fixer.py %filepaths%
python C:\Users\jonora\Documents\cgp_neuron_masters\CGP_Neuron_Masters\statistics_analyzer.py %folderpath% %filepaths%