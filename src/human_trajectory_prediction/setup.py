from setuptools import find_packages, setup

package_name = 'human_trajectory_prediction'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
        'setuptools',
        'pandas',
        'numpy', 
    ],
    zip_safe=True,
    maintainer='even',
    maintainer_email='even@evenlangas.no',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'trajectory_predictor = human_trajectory_prediction.trajectory_predictor:main',
            'trajectory_data_publisher = human_trajectory_prediction.trajectory_data_publisher:main',
        ],
    },
)
