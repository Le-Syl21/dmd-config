# DMD Config Tool

A command-line tool to automatically configure DMD (Dot Matrix Display) and Backglass settings for Visual Pinball tables using DirectB2S files.

## Features

- Automatic DMD area detection in DirectB2S files
- Support for both 2 and 3 screen setups
- Precise template-based rectangle detection
- Configurable screen layouts via YAML file
- Optional image extraction for debugging

## Installation

```bash
pip install dmd-config
```

## Usage

Basic usage:
```bash
dmd-config your_table.directb2s
```

Save extracted images for debugging:
```bash
dmd-config -s your_table.directb2s
```

Use custom screen configuration:
```bash
dmd-config -c custom_config.yaml your_table.directb2s
```

## Configuration

Create a `DMD_config.yaml` file to define your screen layout:

```yaml
screens:
  # Primary display for the playfield
  Playfield:
    id: 1
    size_x: 3840
    size_y: 2160

  # Display for the DMD (set size_x: 0 for 2-screen setup)
  DMD:
    id: 2
    size_x: 1920
    size_y: 1080

  # Display for the backglass
  BackGlass:
    id: 3
    size_x: 2560
    size_y: 1440
```

## Output

The tool generates an INI file with the same name as your DirectB2S file, containing all necessary settings for:
- PinMAME DMD window
- FlexDMD window
- B2S window settings
- Screen positions and dimensions

## Requirements

- Python 3.8 or higher
- OpenCV
- NumPy
- Pillow
- PyYAML

## License

This project is licensed under the GNU General Public License v3 (GPLv3).

## Contributing

Found a bug or want to contribute? Feel free to open an issue or submit a pull request on GitHub.

## Author

Le-Syl21 (sylvain.gargasson@gmail.com)
