from setuptools import find_packages, setup

setup(
    name='connectivity',
    packages=find_packages(),
    version='0.1.0',
    description='Modelling cerebro-cerebellar connectivity using MDTB data',
    author='Maedbh King and Ladan Shahshahani',
    license='MIT',
    entry_points={
    'console_scripts': [
        'transfer-to-savio=connectivity.scripts.data_transfer:to_savio',
        'transfer-from-savio=connectivity.scripts.data_transfer:from_savio'
    ]
    },
)
