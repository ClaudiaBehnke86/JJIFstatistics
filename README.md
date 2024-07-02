# JJIF Statistics

This repository contains code for reading, processing, and displaying statistics related to the Ju-Jitsu International Federation (JJIF) events.

## Overview

This project reads in data from various sources, including sportdata and old sources (JSON), to provide detailed statistics on JJIF events. The code includes functionality for mapping names, performing data analysis, and displaying results using interactive visualizations.

## Features

- **Name Matching**: Uses TF-IDF vectorization and sparse matrix multiplication for effective name matching.
- **Interactive Visualizations**: Utilizes Plotly and Streamlit to create interactive charts and graphs.
- **Data Processing**: Reads and processes data from multiple sources, including CSV and JSON files.
- **User Interface**: Streamlit-based web application with customizable filters and settings.

## Installation

To run this project, you need to have Python installed along with the required libraries. You can install the dependencies using `pip`.

```bash
pip install -r requirements.txt
```

## Usage

To start the Streamlit application, run the following command:

```bash
streamlit run JJIFstats.py
```

## Configuration

The application can be configured through various options:

- **Age Divisions**: Supported age divisions include U16, U18, U21, Adults, U14, U12, U10, U15, and Master. Preselected options are U16, U18, U21, and Adults.
- **Disciplines**: Supported disciplines include Duo, Show, Jiu-Jitsu, Fighting, and Contact. Preselected options are Duo, Show, Jiu-Jitsu, and Fighting.
- **Continents**: Supported continents include Europe, Pan America, Africa, Asia, and Oceania.
- **Event Types**: Supported event types include National Championship, Continental Championship, World Championship, A Class Tournament, B Class Tournament, and World Games / Combat Games. Preselected options are Continental Championship, World Championship, A Class Tournament, B Class Tournament, and World Games / Combat Games.

## Example

To view statistics, navigate to the running Streamlit application in your web browser. You can filter the data by age division, discipline, continent, and event type using the sidebar options.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## Contact

For any questions or support, please contact the JJIF Sport Director at [sportdirector@jjif.org](mailto:sportdirector@jjif.org).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
