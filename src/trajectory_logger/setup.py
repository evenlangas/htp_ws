from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'trajectory_logger'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share',package_name, 'launch'), glob('launch/*.launch.xml')),
        (os.path.join('share',package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='even',
    maintainer_email='even.langas@tvillingfabrikken.no',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'trajectory_logger = trajectory_logger.trajectory_logger:main',
        ],
    },
)
