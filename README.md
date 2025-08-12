# FRACTEL: Framework for Rank Aggregation of CRISPR Tests within ELements
[![CI](https://github.com/Gersbachlab-Bioinformatics/FRACTEL/actions/workflows/ci.yml/badge.svg)](https://github.com/Gersbachlab-Bioinformatics/FRACTEL/actions/workflows/ci.yml)
[![Docker](https://github.com/Gersbachlab-Bioinformatics/FRACTEL/actions/workflows/docker-publish.yml/badge.svg)](https://github.com/Gersbachlab-Bioinformatics/FRACTEL/actions/workflows/docker-publish.yml)
## Overview

FRACTEL is a software designed to find element-level significant discoveries applying a Robust Rank Aggregation (RRA) method to combine individual p-values.

## Installation

To set up the project locally, follow these steps:

1. Install the repository from github:
    ```bash
    pip install --force --no-deps git+https://github.com/Gersbachlab-Bioinformatics/FRACTEL
    ```

## Usage

Run the following command to learn more about the available options:
```bash
$ fractel --help
usage: fractel.py [-h] {run,simulate,interpolate} ...

FRACTEL: Framework for Rank Aggregation of CRISPR Tests within ELements

positional arguments:
  {run,simulate,calibrate}
                        Sub-command to execute
    run                 Run FRACTEL test on a given dataframe with p-values of grouped elements
    simulate            Simulate data for FRACTEL analysis
    calibrate           Calibrate p-values in a data frame based on a background distribution. Currently, the calibration is done by adjusting an empirical cumulative
                        distribution function (ECDF) from the background/reference set and evaluating the observed p-values against it.

options:
  -h, --help            show this help message and exit
```

## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository.
2. Create a new branch:
    ```bash
    git checkout -b feature/your-feature-name
    ```
3. Commit your changes:
    ```bash
    git commit -m "Add your message here"
    ```
4. Push to your branch:
    ```bash
    git push origin feature/your-feature-name
    ```
5. Open a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact

For questions or feedback, please contact alejandro.barrera@duke.edu.
