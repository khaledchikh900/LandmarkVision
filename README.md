# Facial Landmark Collector

## Description
The Facial Landmark Collector is a GUI-based application for collecting and saving facial landmark data using MediaPipe. It provides functionality to save landmark data as either NumPy files or pickle files, with an option to include images.

## Installation

### Conda Environment
Create a conda environment named `LandmarkVision` and install dependencies:
```sh
conda create --name LandmarkVision python=3.8
conda activate LandmarkVision
conda install -c conda-forge numpy pillow
pip install opencv-python mediapipe playsound
```

Ensure all dependencies are installed using the above commands before running the application.

### Jetson Devices
For Jetson devices, ensure that MediaPipe is set up accordingly for accelerated platforms. Follow the instructions provided in the [Jetson MediaPipe setup guide](https://jetson-docs.com/libraries/mediapipe/l4t32.7.1/py3.6.9).

### Configuration
Configure the application using `config.json`:
- `data_path`: Path to save the collected data.
- `actions`: List of actions to be collected.
- `no_sequences`: Number of sequences per action.
- `sequence_length`: Number of frames per sequence.
- `start_folder`: Starting folder index for saving sequences.
- `camera_index`: Index of the camera to use for capturing.
- `min_detection_confidence`: Minimum confidence value for detection.
- `min_tracking_confidence`: Minimum confidence value for tracking.
- `save_images`: Set to `true` to save images along with landmark data. Default is `false` for GDPR compliance.
- `file_format`: File format for saving landmark data (`npy` or `pickle`).

Example `config.json`:
```json
{
    "data_path": "./data",
    "actions": ["wave", "smile"],
    "no_sequences": 30,
    "sequence_length": 30,
    "start_folder": 0,
    "camera_index": 0,
    "min_detection_confidence": 0.5,
    "min_tracking_confidence": 0.5,
    "save_images": false,  
    "file_format": "pickle"
}
```

## Usage

Run the application:
```sh
python3 main.py
```

### UI Guide
1. **Select Action**: Choose the action to collect from the dropdown menu.
2. **Start Collection**: Begin collecting data for the selected action.
3. **Pause/Resume Collection**: Pause or resume the data collection.
4. **Save Data**: Save the collected data to the specified path.
5. **Stop Collection**: Stop the data collection process.
6. **Reset**: Reset the application state.
7. **Exit**: Close the application.

### Data Saving
- **Landmark Data**: Saved in `npy` or `pickle` format based on `file_format` setting.
- **Images**: Saved only if `save_images` is set to `true` in `config.json`.

## GDPR Compliance
By default, the application does not save images to ensure compliance with GDPR. Users can enable image saving by setting `save_images` to `true` in `config.json`.

## License
[MIT License](LICENSE)

## Authors
- [Khaled Chikh](mailto:khaledchikh@unimore.it)
