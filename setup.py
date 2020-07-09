import setuptools


setuptools.setup(
    name="tjevents",
    version="0.1",
    author="Kai Chen",
    author_email="14sdck@tongji.edu.cn",
    description="An event camera data processing library from Institute of Intelligent Vehicle at Tongji University",
    packages=setuptools.find_packages(exclude=('config', 'tools')),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
