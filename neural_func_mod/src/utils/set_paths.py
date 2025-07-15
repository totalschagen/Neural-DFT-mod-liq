import os

def set_paths():
    """
    Set the paths for the data directory, density profiles directory, and neural function directory.
    These paths can be adjusted based on the environment or user preferences.
    """
    
    data_dir = os.getenv("NEURAL_FUNC_DATA_DIR", "/scratch/c7051233/Neural_functional_project")

    density_profiles_dir = os.path.join(data_dir, "data_generation/Density_profiles")

    neural_func_dir = os.path.join(data_dir, "neural_func")
    return data_dir, density_profiles_dir, neural_func_dir
