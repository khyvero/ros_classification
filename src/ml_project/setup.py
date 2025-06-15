from setuptools import find_packages, setup
from glob import glob

package_name = 'ml_project'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    package_data={
        'ml_project': ['data/*.csv']
    },
    data_files = [
        # install CSVs into share/ml_project/data/
        ('share/{0}/data'.format(package_name), glob('data/*.csv')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Veronica',
    maintainer_email='khyveron@gmail.com',
    description='Classification models',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ml_controller = ml_project.ml_controller:main'
        ],
    },
)
