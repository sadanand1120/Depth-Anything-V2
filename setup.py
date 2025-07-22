from setuptools import setup, find_packages

setup(
    name='depthany2',
    version='0.1.0',
    description='DepthAny2 minimal package for depth prediction and point cloud generation',
    author='Sadanand Modak',
    author_email='modaksada11@gmail.com',
    packages=find_packages(include=['depthany2', 'depthany2.*']),
    install_requires=[
        'numpy',
        'opencv-contrib-python',
        'pyyaml',
        'matplotlib',
        'pillow',
        'pyvista>=0.44.0',  # Primary visualization library
    ],
    extras_require={
        'pcd_support': ['open3d>=0.16.0'],  # Optional for PCD file I/O only
    },
    python_requires='>=3.7',
) 