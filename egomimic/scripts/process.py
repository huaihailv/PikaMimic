import os
import h5py

def summarize_hdf5_files(input_folder: str, output_file: str):
    """
    Summarize all HDF5 files in the given folder into a single HDF5 file.

    :param input_folder: Path to the folder containing the episode HDF5 files.
    :param output_file: Path to the output summary HDF5 file.
    """
    # Create the output HDF5 file
    with h5py.File(output_file, 'w') as summary_h5:
        summary_group = summary_h5.create_group('data')  # Top-level group for episodes

        # Iterate over all HDF5 files in the input folder
        for file_name in sorted(os.listdir(input_folder)):
            if file_name.endswith('.hdf5'):
                episode_name = os.path.splitext(file_name)[0]  # Extract name without extension
                episode_path = os.path.join(input_folder, file_name)

                # Open each HDF5 file and copy its content
                with h5py.File(episode_path, 'r') as episode_h5:
                    # Create a subgroup for the current episode in the summary file
                    episode_group = summary_group.create_group(episode_name)
                    copy_hdf5_data(episode_h5, episode_group)

def copy_hdf5_data(source_group, target_group):
    """
    Recursively copy data from one HDF5 group to another.

    :param source_group: The source HDF5 group.
    :param target_group: The target HDF5 group in the summary file.
    """
    for key, item in source_group.items():
        if isinstance(item, h5py.Dataset):
            # Copy datasets
            target_group.create_dataset(key, data=item[...])  # Copy dataset content
        elif isinstance(item, h5py.Group):
            # Recursively copy groups
            new_group = target_group.create_group(key)
            copy_hdf5_data(item, new_group)

# Example usage:
# Replace 'input_folder_path' with the folder containing your HDF5 files.
# Replace 'output_summary_file_path' with the desired output summary file path.
input_folder_path = './input_hdf5_files'
output_summary_file_path = './summary.hdf5'

summarize_hdf5_files(input_folder_path, output_summary_file_path)
