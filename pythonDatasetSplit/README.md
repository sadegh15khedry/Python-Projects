# Python Dataset Split

## Overview

The Python Dataset Split project provides a simple utility for splitting datasets into training, validation, and test sets. This project is useful for machine learning practitioners who need to partition their datasets for training and evaluation purposes. The utility is implemented in Python and provides flexibility in specifying the split ratios.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
  - [Clone the Repository](#clone-the-repository)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Features

- **Dataset Splitting**: Partition datasets into training, validation, and test sets.
- **Custom Split Ratios**: Specify custom split ratios for each partition.
- **Random Sampling**: Perform random sampling to ensure diverse samples in each partition.

## Installation

### Clone the Repository

1. **Clone the repository:**
   ```
   git clone https://github.com/sadegh15khedry/pythonDatasetSplit.git
   cd pythonDatasetSplit
   ```

## Usage

1. **Prepare Dataset**: Prepare your dataset as a CSV or other compatible format.
2. **Run the Script**: Run the Python script to split the dataset into training, validation, and test sets.
   ```python
   python split_dataset.py --input dataset.csv --output_dir output --train_ratio 0.7 --val_ratio 0.2 --test_ratio 0.1
   ```

## Contributing

Contributions are welcome! If you'd like to contribute to the project, please follow these steps:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/YourFeature`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Create a new Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or inquiries, please feel free to open an issue on the GitHub repository.
