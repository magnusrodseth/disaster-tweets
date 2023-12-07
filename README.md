# Data Driven Software

## Running the code

### Prerequisites for running the code

You can install dependencies however you like. However, it is recommended to use `conda`, as it comes with a lot of useful packages pre-installed. Read more about `conda` [here](https://docs.conda.io/en/latest/).

### Installing dependencies

```sh
# Navigate to the project directory
cd it3212

# Create a new conda environment
conda create --name it3212

# Activate the environment
conda activate it3212

# Install dependencies
conda install --file requirements.txt
```

### Executing the code

It is recommended to run the Jupyter Notebook in VSCode. Ensure the `it3212` conda environment is selected as the Python interpreter.

## Generating deliverable reports

### Prerequisites for generating reports

Ensure you have `make` and `pandoc` installed. `make` is used to run the `Makefile`, and `pandoc` is used to convert `.md` to a `.pdf` in a nice format.

```sh
# Install make
brew install make

# Install pandoc
brew install pandoc
```

### Generating the reports

```sh
# Navigate to the project directory
cd it3212

# Generate the reports
make
```

Note that all PDF files are gitignored, so you they will not be added to version control.

## Developer Information

Developed by Haakon Tideman Kanter, [Henrik Skog](https://github.com/henrikskog), Mattis Hembre, Max Gunhamn, Sebastian Sole, and [Magnus Rødseth](https://github.com/magnusrodseth).
