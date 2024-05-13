from setuptools import find_packages, setup

package_name = 'detect_cylinders'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='lb',
    maintainer_email='lb2448@student.uni-lj.si',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'detect_cyliders_script = detect_cylinders.detect_cyliders_script:main'
        ],
    },
)
