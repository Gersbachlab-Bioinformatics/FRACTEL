from setuptools import setup, find_packages

setup(
    name="fractel",
    version="0.1.0",
    author="Alejandro Barrera",
    author_email="alejandro.barrera@duke.edu",
    description="FRACTEL: Framework for Rank Aggregation of CRISPR Tests within ELements",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/fractel",
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=[
        line.strip() for line in open("requirements.txt").readlines() if line.strip() and not line.startswith("#")
    ],
    entry_points={
        "console_scripts": [
            "fractel=fractel.fractel:main",
        ],
    },
)
