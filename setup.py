import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="NaroNet", # Replace with your own username
    version="0.0.11",
    author="Daniel Jiménez-Sánchez",
    author_email="danijimnzs@gmail.com",
    description="NaroNet: discovering tumor microenvironment elements from multiplex imaging.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/djimenezsanchez/NaroNet",
    project_urls={
        "Bug Tracker": "https://github.com/djimenezsanchez/NaroNet/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
		"License :: OSI Approved :: BSD License",
    ],
	license='BSD 3-Clause License',
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.7",
    install_requires = [
        'matplotlib>=3.2.1','pandas>=1.1.5','seaborn>=0.11.0',
        'scikit-learn','scikit-image','Pillow',
        'scipy>=1.5.4','numpy>=1.18.2','tifffile>=2020.2.16',
        'pycox>=0.2.0','sklearn_pandas>=2.0.3','torchtuples>=0.2.0',
        'opencv-python>=4.2.0','tqdm>=4.50.2','ray[tune]',
        'xlsxwriter>=1.1.5','imgaug>=0.4.0',
        'xlrd>=1.2.0','tensorboard>=1.14.0',
        'argparse>=1.1','hyperopt>=0.2.3','tensorflow>=1.14.0',
        'torch>=1.4.0','imblearn','imagecodecs'
    ]
)


# Upload your package to PyPi
# Now, we create a source distribution with the following command:
# python setup.py sdist

# To upload to pypi test
# twine upload --repository testpypi dist/*
# To upload to pypi
# twine upload dist/*
# You will be asked to provide your username and password. Provide the credentials you used to register to PyPi earlier.