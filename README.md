# FRACTEL: Framework for Rank Aggregation of CRISPR Tests within ELements!
[![CI](https://github.com/Gersbachlab-Bioinformatics/FRACTEL/actions/workflows/ci.yml/badge.svg)](https://github.com/Gersbachlab-Bioinformatics/FRACTEL/actions/workflows/ci.yml)

## Overview

FRACTEL is a software designed to find element-level significant discoveies applying a Robust Rank Aggregation (RRA) method to combine individual p-values.

## Installation

To set up the project locally, follow these steps:

1. Install the repository from github:
    ```bash
    pip install git+https://github.com/Gersbachlab-Bioinformatics/FRACTEL
    ```

## Usage

Run the following command to start the application:
```bash
fractel run --help
usage: fractel.py [-h] {run,simulate} ...

Run FRACTEL test or simulate data

positional arguments:
  {run,simulate}  Sub-command to execute
    run           Run FRACTEL test on a given dataframe with p-values of grouped elements
    simulate      Simulate data for FRACTEL analysis

options:
  -h, --help      show this help message and exit
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
