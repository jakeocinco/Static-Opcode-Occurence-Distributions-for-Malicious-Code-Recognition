
DISTRIBUTION_SAMPLE_PATH = f"/Volumes/T7/pe_machine_learning_set/pe-machine-learning-dataset/op_code_distributions_samples/"

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
    # {
    #     'name': 'VirusShare00005',
    #     'executable_directory': "/Volumes/T7/VirusShare/executables/VirusShare00005",
    #     'label': 'infected'
    # }
]