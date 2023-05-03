
DISTRIBUTION_SAMPLE_PATH = f"/Volumes/T7/pe_machine_learning_set/pe-machine-learning-dataset/op_code_distributions_samples/"
RESULTS_BASE_PATH = f"/Volumes/T7/pe_machine_learning_set/pe-machine-learning-dataset/results/"

# ADD SAVED DATA PATH

TRAINING_SAMPLES = [
    {
        'name': 'pe-machine-learning-dataset',
        'label_csv': '/Volumes/T7/pe_machine_learning_set/pe-machine-learning-dataset/samples.csv',
        'executable_directory': '/Volumes/T7/pe_machine_learning_set/pe-machine-learning-dataset/samples',
        'op_code_list_directory': '/Volumes/T7/pe_machine_learning_set/pe-machine-learning-dataset/op_code_samples',
        'benign_identifier': 'Whitelist',
        'malicious_identifier': 'Blacklist'
    }
]

TESTING_SAMPLES = [
    {
        'name': 'VirusShare00000',
        'executable_directory': "/Volumes/T7/VirusShare/executables/VirusShare_00000",
        'op_code_list_directory': "/Volumes/T7/VirusShare/VirusShare_00000/op_code_samples",
        'label': 'infected'
    },
    {
        'name': 'VirusShare00005',
        'executable_directory': "/Volumes/T7/VirusShare/executables/VirusShare_00005",
        'op_code_list_directory': "/Volumes/T7/VirusShare/VirusShare_00005/op_code_samples",
        'label': 'infected'
    },
    {
        'name': 'VirusShare00451',
        'executable_directory': "/Volumes/T7/VirusShare/executables/VirusShare_00451",
        'op_code_list_directory': "/Volumes/T7/VirusShare/VirusShare_00451/op_code_samples",
        'label': 'infected'
    },
    {
        'name': 'MacOS',
        'label_csv': '/Volumes/T7/MacOS/MacOSExecutableList.csv',
        'executable_directory': '',
        'op_code_list_directory': "/Volumes/T7/MacOS/op_code_samples",
        'benign_identifier': 'clean',
        'malicious_identifier': 'infected'
    }
]